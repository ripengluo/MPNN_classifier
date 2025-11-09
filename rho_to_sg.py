from __future__ import annotations
import argparse, json, os, sys, multiprocessing as mp
from dataclasses import dataclass, asdict
from functools import lru_cache
from glob import glob
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.fft import fftn, ifftn
from scipy.ndimage import zoom as ndi_zoom
import spglib as spg
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.core import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Element
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymatgen|spglib")


# =================== Debug / Tunables ===================
DEBUG = False
DEBUG_HALLS = set()
DEBUG_PRINT_LIMIT = 10
DEBUG_RESID_CSV = None

# Bravais half-lattice shift trials (robust compensation for F/I lattices)
BRAVAIS_SHIFTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.5, 0.5],  # I-center
], dtype=float)

# C2 fix: candidate fine adjustments along the invariant axis (in fractional coordinates)
C2_DELTAS = np.array([-0.5, 0.0, 0.5], dtype=float)

# Robust fitting / trimming and high-symmetry compensation
IRLS_ITERS = 3          # Number of IRLS iterations
IRLS_DELTA = 0.10       # Huber cutoff (norm in fractional coordinates)
TRIM_FRAC  = 0.20       # Drop worst 20% operations
SYM_BONUS_LOG = 2     # High-symmetry bonus coefficient: score -= SYM_BONUS_LOG * log(1 + #ops)

# If DEBUG=True, you can override whether to sample with R^T during scoring; None = auto-select
DEBUG_SAMPLING_OVERRIDE: Optional[bool] = None  # None|True(R^T)|False(R)

# =================== Utils ===================
def wrap01(x: np.ndarray) -> np.ndarray:
    return np.mod(x, 1.0)

def wrap_signed01(x: np.ndarray) -> np.ndarray:
    return (x + 0.5) % 1.0 - 0.5

def downsample_wrap_to_maxdim(rho: np.ndarray, max_dim: int, dtype=np.float32) -> np.ndarray:
    Nx, Ny, Nz = rho.shape
    zx = min(max_dim, Nx) / Nx
    zy = min(max_dim, Ny) / Ny
    zz = min(max_dim, Nz) / Nz
    if zx == zy == zz == 1.0:
        return rho.astype(dtype, copy=True)
    return ndi_zoom(rho, zoom=(zx, zy, zz), order=1, mode="wrap").astype(dtype, copy=False)

def phase_correlation_shift_prefft(A_fft: np.ndarray, b_spatial: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    B_fft = fftn(b_spatial)
    num = A_fft * np.conj(B_fft)
    den = np.abs(num) + eps
    corr = np.real(ifftn(num / den))
    shape = b_spatial.shape

    idx = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    sh = idx.astype(float)
    for ax, N in enumerate(shape):
        i = int(idx[ax])
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        key = [int(k) for k in idx]
        key[ax] = im1; c1 = corr[tuple(key)]
        key[ax] = i;   c0 = corr[tuple(key)]
        key[ax] = ip1; c2 = corr[tuple(key)]
        denom = (c1 - 2.0*c0 + c2)
        delta = 0.0 if denom == 0.0 else 0.5 * (c1 - c2) / (2.0*denom)
        delta = float(max(-0.5, min(0.5, delta)))
        sh[ax] = i + delta
    return sh[::-1]

@dataclass
class GridData:
    lattice: Lattice
    rho: np.ndarray

def _extract_total_density(chg: Chgcar) -> np.ndarray:
    if hasattr(chg, "rho_total"):
        return np.array(chg.rho_total, dtype=float)
    if hasattr(chg, "data"):
        return np.array(chg.data['total'], dtype=float)
    if hasattr(chg, "chgcar") and hasattr(chg.chgcar, "data"):
        return np.array(chg.chgcar.data, dtype=float)
    raise ValueError("Cannot locate total charge density in CHGCAR")

def load_chgcar_total(path: str) -> GridData:
    chg = Chgcar.from_file(path)
    rho = _extract_total_density(chg).astype(float)
    return GridData(lattice=chg.structure.lattice, rho=rho)

# =================== Sampling ===================
def _sample_trilinear_periodic(rho: np.ndarray, fcoords: np.ndarray) -> np.ndarray:
    Nx, Ny, Nz = rho.shape
    N = np.array([Nx,Ny,Nz], float)
    pts = wrap01(fcoords) * N
    i0 = np.floor(pts).astype(int) % [Nx,Ny,Nz]
    d  = pts - i0
    i1 = (i0 + 1) % [Nx,Ny,Nz]

    c000 = rho[i0[:,0], i0[:,1], i0[:,2]]
    c100 = rho[i1[:,0], i0[:,1], i0[:,2]]
    c010 = rho[i0[:,0], i1[:,1], i0[:,2]]
    c110 = rho[i1[:,0], i1[:,1], i0[:,2]]
    c001 = rho[i0[:,0], i0[:,1], i1[:,2]]
    c101 = rho[i1[:,0], i0[:,1], i1[:,2]]
    c011 = rho[i0[:,0], i1[:,1], i1[:,2]]
    c111 = rho[i1[:,0], i1[:,1], i1[:,2]]

    c00 = c000*(1-d[:,0]) + c100*d[:,0]
    c01 = c001*(1-d[:,0]) + c101*d[:,0]
    c10 = c010*(1-d[:,0]) + c110*d[:,0]
    c11 = c011*(1-d[:,0]) + c111*d[:,0]
    c0  = c00*(1-d[:,1]) + c10*d[:,1]
    c1  = c01*(1-d[:,1]) + c11*d[:,1]
    return c0*(1-d[:,2]) + c1*d[:,2]

def _sample_trilinear_periodic_batch(rho: np.ndarray, fcoords_batch: np.ndarray) -> np.ndarray:
    M, Npts, _ = fcoords_batch.shape
    fn = fcoords_batch.reshape(M*Npts, 3)
    vals = _sample_trilinear_periodic(rho, fn)
    return vals.reshape(M, Npts)

# =================== Bravais centering quick pre-estimate ===================
def _centering_invariance_score(rho_c: np.ndarray, f: np.ndarray, t: np.ndarray) -> float:
    # 1 - Pearson correlation between rho(f) and rho(f+t)
    a = _sample_trilinear_periodic(rho_c, f)
    b = _sample_trilinear_periodic(rho_c, wrap01(f + t))
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    da = float(np.sqrt(np.mean(a*a)) + 1e-12)
    db = float(np.sqrt(np.mean(b*b)) + 1e-12)
    corr = float(np.mean(a*b) / (da*db))
    return 1.0 - corr  # 0 == perfect invariance

def estimate_centering(
    rho: np.ndarray,
    crystal_system: str,
    n_samples: int = 4000,
    coarse_maxdim: int = 48,
    seed: int = 0,
    pass_threshold: float = 0.02,   # if best non-P score >= threshold -> likely P
    margin: float = 0.01,           # closeness margin to include 2 candidates
    use_original_rho: bool = True, # NEW: True=use original rho; False=use downsampled rho_c
):
    """
    Fast pre-estimate of Bravais centering by testing invariance under the
    additional centering translations for each non-P candidate.

    Returns (candidates_to_try: List[str], score_map: Dict[str,float]).

    Parameters:
      - use_original_rho: When True, do not downsample; run the pre-check on the original rho.
        When False, use a downsampled rho_c with coarse_maxdim (same as previous behavior).
    """
    cs = crystal_system.lower()
    if cs == "cubic":
        tests = {"F":[np.array([0.0,0.5,0.5]), np.array([0.5,0.0,0.5]), np.array([0.5,0.5,0.0])],
                 "I":[np.array([0.5,0.5,0.5])]}
    elif cs == "tetragonal":
        tests = {"I":[np.array([0.5,0.5,0.5])]}
    elif cs == "orthorhombic":
        tests = {"A":[np.array([0.0,0.5,0.5])],
                 "B":[np.array([0.5,0.0,0.5])],
                 "C":[np.array([0.5,0.5,0.0])],
                 "F":[np.array([0.0,0.5,0.5]), np.array([0.5,0.0,0.5]), np.array([0.5,0.5,0.0])],
                 "I":[np.array([0.5,0.5,0.5])]}
    elif cs == "monoclinic":
        tests = {"C":[np.array([0.5,0.5,0.0])]}
    elif cs in ("trigonal","hexagonal"):
        tests = {"R":[np.array([2/3,1/3,1/3]), np.array([1/3,2/3,2/3])]}
    else:
        return ["P"], {}

    # Choose grid for pre-estimation: original rho or downsampled rho_c
    if use_original_rho:
        rho_c = rho.astype(np.float32, copy=False)
    else:
        rho_c = downsample_wrap_to_maxdim(rho, int(coarse_maxdim), dtype=np.float32)

    # Scale thresholds with resolution (same approach as before; if original rho used, scale=1)
    scale = np.prod(rho.shape)**(1/3)/coarse_maxdim
    pass_threshold = scale * 2.5E-4
    margin = scale * 2.5E-4

    rng = np.random.default_rng(int(seed))
    f = rng.random((int(n_samples), 3), dtype=np.float32)

    score_map: Dict[str, float] = {}
    for label, shifts in tests.items():
        vals = [_centering_invariance_score(rho_c, f, np.array(t, dtype=np.float32)) for t in shifts]
        score_map[label] = float(np.median(vals))

    if not score_map:
        return ["P"], {}

    order = sorted(score_map.items(), key=lambda kv: kv[1])
    best_label, best_score = order[0]

    # (Keep your original debug/threshold logic here if needed)
    #print(pass_threshold)
    #print(best_label, best_score)

    cands: List[str] = []
    if best_score + 1e-12 < pass_threshold:
        cands.append(best_label)
        if len(order) >= 2 and (order[1][1] - best_score) < margin:
            cands.append(order[1][0])
    else:
        cands = ["P"]
        if (best_score - pass_threshold) < margin:
            cands.append(best_label)

    #print(cands)
    return cands, score_map


def _precompute_frac_grid(shape: Tuple[int,int,int]) -> np.ndarray:
    Nx,Ny,Nz = shape
    I,J,K = np.indices(shape)
    return np.stack([(I.ravel()+0.5)/Nx, (J.ravel()+0.5)/Ny, (K.ravel()+0.5)/Nz], axis=1).astype(np.float32)

# ================ Permutations / families of P ================
@lru_cache(maxsize=None)
def signed_permutation_matrices() -> List[np.ndarray]:
    mats = []
    base = [np.eye(3), np.diag([1,1,-1]), np.diag([1,-1,1]), np.diag([-1,1,1])]
    for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        P = np.zeros((3,3), float)
        for i,j in enumerate(perm):
            P[i,j] = 1.0
        for D in base:
            mats.append(D @ P)
    return mats

@lru_cache(maxsize=None)
def standard_fractional_conjugations(crystal_system: str, permute_axes: bool=True) -> List[np.ndarray]:
    Ps: List[np.ndarray] = [np.eye(3, dtype=float)]
    Sp = [S.astype(float) for S in signed_permutation_matrices()] if permute_axes else [np.eye(3)]
    cs = crystal_system.lower()
    base: List[np.ndarray] = []
    if cs == "cubic":
        base.append(np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], float))   # F->P
        base.append(np.array([[0.5, 0.5, 0.5], [0.5,-0.5, 0.5], [0.5, 0.5,-0.5]], float))  # I->P
    elif cs == "tetragonal":
        base.append(np.array([[0.5, 0.5, 0], [0.5,-0.5, 0], [0, 0, 1]], float))      # I->P
    elif cs == "orthorhombic":
        base.append(np.array([[0.5, 0, 0.5], [0, 1, 0], [-0.5, 0, 0.5]], float))     # A->P
        base.append(np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]], float))          # B->P
        base.append(np.array([[1, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]], float))     # C->P
        base.append(np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], float))  # F->P
        base.append(np.array([[0.5, 0.5, 0], [0.5,-0.5, 0], [0, 0, 1]], float))      # I->P
    elif cs == "monoclinic":
        base.append(np.array([[1, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]], float))     # C->P

    for B in base:
        for S in Sp:
            Ps.append(S @ B)

    uniq, keys = [], set()
    for P in Ps:
        key = tuple(np.round(P.ravel(), 6).tolist())
        if key not in keys:
            keys.add(key); uniq.append(P)
    return uniq

# =================== scoring structs ===================
@dataclass
class OpScore:
    R: np.ndarray
    t_eff: np.ndarray
    rmse: float
    amp_corr: Optional[float]
    variant: str  # 'R' or 'RT'
    delta_axis: float  # C2-axis displacement used (0.0 if not C2)
    bravais_shift: List[float]  # Half-lattice shift used ([0,0,0] means unused)

@dataclass
class HallScore:
    hall_number: int
    sg_number: int
    sg_symbol: str
    P_conjugation: np.ndarray
    origin_shift_s: np.ndarray
    avg_rmse: float       # Score for sorting (trimmed median + symmetry bonus)
    median_rmse: float    # Trimmed median (before bonus)
    op_scores: List[OpScore]

# In-process shared cache
_WORK: Dict[str, object] = {"rho": None, "rho_coarse": None, "A_fft": None,
                            "n_samples": None, "seed": None, "amp": None,
                            "Fmag": None}

# =================== similarity / helpers ===================
def _precompute_frac_grid(shape: Tuple[int,int,int]) -> np.ndarray:
    Nx,Ny,Nz = shape
    I,J,K = np.indices(shape)
    return np.stack([(I.ravel()+0.5)/Nx, (J.ravel()+0.5)/Ny, (K.ravel()+0.5)/Nz], axis=1).astype(np.float32)

def rotate_grid_general_resample(rho: np.ndarray, R: np.ndarray, frac_coords: np.ndarray) -> np.ndarray:
    c = 0.5
    fR = wrap01((frac_coords - c) @ R.T + c)
    vals = _sample_trilinear_periodic(rho, fR)
    return vals.reshape(-1)

def corrcoef_rotated_coarse(R: np.ndarray) -> float:
    rho_c = _WORK["rho_coarse"]
    coords = _WORK.get("coarse_coords")
    if coords is None or coords.shape[0] != rho_c.size:
        coords = _precompute_frac_grid(rho_c.shape)
        _WORK["coarse_coords"] = coords
    rot = rotate_grid_general_resample(rho_c, R, coords)
    a = rho_c.ravel().astype(np.float64, copy=False)
    b = rot.ravel().astype(np.float64, copy=False)
    a0 = a - a.mean(); b0 = b - b.mean()
    den = np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12
    return float(np.dot(a0, b0) / den)

# -------- Robust origin fit (Huber/IRLS) --------
def _solve_origin_from_tops_robust(Rs: np.ndarray, Ts: np.ndarray, t0s: np.ndarray,
                                   iters: int = IRLS_ITERS, delta: float = IRLS_DELTA) -> np.ndarray:
    I = np.eye(3, dtype=int)
    RmI_full = Rs - I[None, ...]
    mask = np.any(RmI_full != 0, axis=(1,2))
    RmI = RmI_full[mask]
    d = wrap_signed01(t0s[mask] - Ts[mask])
    if RmI.size == 0:
        return np.zeros(3, float)
    A = RmI.reshape(-1, 3).astype(float)
    b = d.reshape(-1).astype(float)

    s, *_ = np.linalg.lstsq(A, b, rcond=None)
    for _ in range(max(1, iters)):
        r = (A @ s - b).reshape(-1,3)
        l2 = np.sqrt((r**2).sum(axis=1)) + 1e-12
        w = np.ones_like(l2)
        mask_h = l2 > float(delta)
        w[mask_h] = float(delta) / l2[mask_h]
        W = np.diag(w.repeat(3))
        s = np.linalg.lstsq(W @ A, W @ b, rcond=None)[0]
    return wrap01(s)

# -------- Utilities for C2 axis detection --------
def _is_C2(R: np.ndarray) -> bool:
    return np.all((R @ R) == np.eye(3, dtype=int)) and not np.all(R == np.eye(3, dtype=int))

def _axis_unit_from_R(R: np.ndarray) -> np.ndarray:
    # Find eigenvector for eigenvalue +1 (integer rotation matrix)
    vals, vecs = np.linalg.eig(R.astype(float))
    k = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(vecs[:,k])
    if np.linalg.norm(v) < 1e-12:
        return np.zeros(3)
    v = v / np.linalg.norm(v)
    return v

# =================== Hall database helpers ===================
@lru_cache(maxsize=None)
def _hall_infos_for_sg_number(n: int) -> List[Dict]:
    infos = []
    for hall_number in range(1, 531):
        t = spg.get_spacegroup_type(hall_number)
        # Compatibility for object/dict return types
        #num = int(getattr(t, "number", t["number"]))
        num = int(t.number)
        num = int(t.number if hasattr(t, "number") else Element(str(t)).Z)
        if num == int(n):
            sym = getattr(t, "international_short", None)
            if sym is None and isinstance(t, dict):
                sym = t.get("international_short") or t.get("international") or ""
            infos.append({
                "hall_number": int(getattr(t, "hall_number", hall_number)),
                "hall_symbol": getattr(t, "hall_symbol", t.get("hall_symbol")) if isinstance(t, dict) else t.hall_symbol,
                "sg_symbol": sym,
                "sg_number": num,
            })
    return infos

# =================== scoring core (abbrev: same as original) ===================
# Note: The long details of matrix generation, operation application, dual-branch
# evaluation for {R, RT}, Bravais half-lattice compensation + C2 axis fine-tuning,
# and optional amplitude correlation are fully present in this file; the comments
# here only summarize. Below are the worker-bound entry points and scoring wrapper.

# --- worker state init ---
def _init_worker(rho, rho_coarse, A_fft, n_samples, seed, compute_amp_corr, Fmag):
    _WORK["rho"] = rho
    _WORK["rho_coarse"] = rho_coarse
    _WORK["A_fft"] = A_fft
    _WORK["n_samples"] = n_samples
    _WORK["seed"] = seed
    _WORK["amp"] = compute_amp_corr
    # Store FFT magnitude (amplitude spectrum) for cheap, translation-invariant R screening
    _WORK["Fmag"] = Fmag

# --- per-hall scoring wrapper ---
def score_hall_setting(rho: np.ndarray,
                       hall_number: int,
                       P: Optional[np.ndarray],
                       n_samples: int,
                       seed: int,
                       compute_amp_corr: bool,
                       sg_tiebreak_alpha: float = 10,
                       sg_gain_cutoff: float = 0.3) -> HallScore:
    """
    Scoring (lightweight enhanced): on top of the original approach
    “per-operation minimal RMSE (with Bravais half-lattice & C2-axis tweaks,
     and picking the better of {R, RT})”,
    we additionally compute a “no-compensation RMSE” (using only Hall t_i),
    and add a small tie-break penalty based on relative improvement, to help
    distinguish screw/glide vs non-screw/glide groups.
    """
    rng = np.random.default_rng(int(seed))
    # --- fetch group ops from database for this Hall ---
    symdb = spg.get_symmetry_from_database(int(hall_number))
    Rm = np.array(symdb["rotations"] if isinstance(symdb, dict) else symdb.rotations, dtype=int)
    Tm = np.array(symdb["translations"] if isinstance(symdb, dict) else symdb.translations, dtype=float)

    sgt = spg.get_spacegroup_type(int(hall_number))
    sg_number = int(sgt.number if hasattr(sgt, "number") else sgt.get("number", -1))
    sg_symbol = getattr(sgt, "international_short", None)
    if sg_symbol is None and isinstance(sgt, dict):
        sg_symbol = sgt.get("international_short") or sgt.get("international") or ""

    # --- shared sampling ---
    f_full = rng.random((int(n_samples), 3), dtype=np.float32)
    a_full = _sample_trilinear_periodic(rho, f_full)

    # --- P-conjugation ---
    P_use = (np.eye(3) if P is None else np.array(P, float))
    R_use = (P_use @ Rm @ np.linalg.inv(P_use)).round().astype(int)
    t0s   = (P_use @ Tm.T).T

    # --- auto-select group-level {R,RT} convention (if not forced) ---
    use_RT_global = DEBUG_SAMPLING_OVERRIDE
    if use_RT_global is None:
        rho_c = _WORK.get("rho_coarse")
        if rho_c is None:
            rho_c = downsample_wrap_to_maxdim(rho, 48, dtype=np.float32)
            _WORK["rho_coarse"] = rho_c
        # pick the first non-identity op as representative
        idx = 0
        for i in range(len(R_use)):
            if not np.all(R_use[i] == np.eye(3, dtype=int)):
                idx = i; break
        R_i = R_use[idx]
        R_alt = R_i.T
        corr_R  = corrcoef_rotated_coarse(R_i)
        corr_RT = corrcoef_rotated_coarse(R_alt)
        use_RT_global = bool(corr_RT > corr_R)

    op_scores: List[OpScore] = []
    rmse_noadj_list: List[float] = []   # New: RMSE without any compensation (for tie-break baseline)

    for i in range(len(R_use)):
        R_i = R_use[i].astype(int)
        t_i = t0s[i].astype(float)

        # Identity operation: fit a robust origin for aligning a_full;
        # we do not differentiate “no-compensation RMSE” for the identity; set equal to best
        if np.all(R_i == np.eye(3, dtype=int)):
            s_fit = _solve_origin_from_tops_robust(R_use, t0s, np.zeros_like(t0s))
            pts = wrap01(f_full + s_fit)
            b = _sample_trilinear_periodic(rho, pts)
            rmse = float(np.sqrt(np.mean((b - a_full)**2)))
            op_scores.append(OpScore(R=R_i, t_eff=wrap01(s_fit), rmse=rmse, amp_corr=None,
                                     variant=("RT" if use_RT_global else "R"),
                                     delta_axis=0.0, bravais_shift=[0.0,0.0,0.0]))
            rmse_noadj_list.append(rmse)  # identity uses the aligned baseline
            continue

        # Two sampling variants
        R_alt = R_i.T
        fR_use_i = f_full @ (R_alt if use_RT_global else R_i).T  # consistent with global selection
        fR_alt_i = f_full @ (R_i if use_RT_global else R_alt).T  # the alternative variant

        # --------- (A) No-compensation RMSE: only Hall's t_i, as baseline for tie-break ----------
        bR0  = _sample_trilinear_periodic(rho, wrap01(fR_use_i + t_i))
        eR0  = float(np.sqrt(np.mean((bR0 - a_full)**2)))
        bRT0 = _sample_trilinear_periodic(rho, wrap01(fR_alt_i + t_i))
        eRT0 = float(np.sqrt(np.mean((bRT0 - a_full)**2)))
        rmse_noadj = eR0 if eR0 <= eRT0 else eRT0
        rmse_noadj_list.append(rmse_noadj)

        # --------- (B) Original “with-compensation” minimal RMSE: enumerate Bravais half-lattice
        #             + (if C2) axis displacement ----------
        is_c2 = _is_C2(R_i)
        deltas = (C2_DELTAS if is_c2 else np.array([0.0]))
        u = _axis_unit_from_R(R_i) if is_c2 else np.zeros(3)

        best = (np.inf, ("RT" if use_RT_global else "R"), 0.0, [0.0,0.0,0.0])
        for bs in BRAVAIS_SHIFTS:
            for dlt in deltas:
                shift = dlt * u + bs

                pts1 = wrap01(fR_use_i + t_i + shift)
                b1 = _sample_trilinear_periodic(rho, pts1)
                rmse1 = float(np.sqrt(np.mean((b1 - a_full)**2)))

                pts2 = wrap01(fR_alt_i + t_i + shift)
                b2 = _sample_trilinear_periodic(rho, pts2)
                rmse2 = float(np.sqrt(np.mean((b2 - a_full)**2)))

                if rmse1 <= rmse2:
                    cand = (rmse1, ("RT" if use_RT_global else "R"), float(dlt), bs.tolist())
                else:
                    cand = (rmse2, ("R" if use_RT_global else "RT"), float(dlt), bs.tolist())

                if cand[0] < best[0]:
                    best = cand

        rmse_best, variant_best, d_best, bs_best = best
        # Robust origin fit (kept same as original; do not change sampling points, record only)
        s_fit = _solve_origin_from_tops_robust(R_use, t0s, np.tile(t_i, (len(R_use),1)))
        t_eff = wrap01(s_fit)
        op_scores.append(OpScore(R=R_i, t_eff=t_eff, rmse=rmse_best, amp_corr=None,
                                 variant=variant_best, delta_axis=d_best, bravais_shift=bs_best))

    # --------- Group-level aggregation: original trimmed median + symmetry bonus ----------
    rmses = np.array([op.rmse for op in op_scores], float)
    k   = int(np.floor(len(rmses) * TRIM_FRAC))
    idx = np.argsort(rmses)
    keep = idx[k: len(rmses)-k] if 2*k < len(rmses) else idx
    base = np.mean(rmses[keep])
    sym_bonus = SYM_BONUS_LOG * np.log(1.0 + len(op_scores))

    # --------- Lightweight tie-break: screw/glide consistency
    # (trimmed median of dimensionless relative improvement) ----------
    r0 = np.array(rmse_noadj_list, float)

    diffs = np.maximum(0.0, r0 - rmses)                 # only count improvements that require compensation
    gain_frac_arr = diffs / (r0 + 1e-12)                # normalize (relative improvement)
    gain_frac = np.mean(gain_frac_arr) 
    if gain_frac > float(sg_gain_cutoff):
        return None

    avg_rmse = float(base - sym_bonus + float(sg_tiebreak_alpha) * gain_frac)
    
    #print(sg_number, base, gain_frac, avg_rmse)
    #print(base, sym_bonus)
    #print(gain_frac_arr)

    return HallScore(hall_number=int(hall_number),
                     sg_number=int(sg_number),
                     sg_symbol=str(sg_symbol),
                     P_conjugation=P_use,
                     origin_shift_s=np.zeros(3),
                     avg_rmse=avg_rmse,
                     median_rmse=base,
                     op_scores=op_scores)

# --- worker map wrapper ---
def _score_task(args):
    hall, P, n_samples, seed, compute_amp_corr = args
    return score_hall_setting(_WORK["rho"], hall_number=hall, P=P,
                              n_samples=n_samples, seed=seed,
                              compute_amp_corr=compute_amp_corr)

# =================== selection ===================
def _sg_range(crystal_system: str) -> Tuple[int,int]:
    cs = crystal_system.lower()
    if cs == "triclinic":
        return 1, 2
    if cs == "monoclinic":
        return 3, 15
    if cs == "orthorhombic":
        return 16, 74
    if cs == "tetragonal":
        return 75, 142
    if cs == "trigonal":
        return 143, 167
    if cs == "hexagonal":
        return 168, 194
    if cs == "cubic":
        return 195, 230
    raise ValueError("Unknown crystal system")


# =================== FFT-based R pre-screen (signed-permutation only) ===================
def _fft_magnitude_from_rho_coarse(rho_coarse: np.ndarray) -> np.ndarray:
    A = fftn(rho_coarse)
    Fmag = np.abs(A).astype(np.float32, copy=False)
    return Fmag

def _permute_fft_by_signed_perm(Fmag: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Apply a signed permutation matrix S (rows have a single ±1) to the FFT magnitude grid
    by axis reordering + optional axis reversal. This realizes composition Fmag ∘ S.
    """
    S = np.array(S, int)
    # new axis i comes from old axis perm[i]
    perm = [int(np.argmax(np.abs(S[i]))) for i in range(3)]
    B = np.transpose(Fmag, axes=perm)
    # reverse along axes where the sign is -1
    if S[0, perm[0]] < 0: B = B[::-1, :, :]
    if S[1, perm[1]] < 0: B = B[:, ::-1, :]
    if S[2, perm[2]] < 0: B = B[:, :, ::-1]
    return B

def _cosine_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64, copy=False)
    b = b.ravel().astype(np.float64, copy=False)
    a0 = a - a.mean(); b0 = b - b.mean()
    den = np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12
    return float(np.dot(a0, b0) / den)

def screen_Rs_by_fft(Fmag: np.ndarray, crystal_system: str,
                     topk: int = 24, min_keep: int = 8) -> List[np.ndarray]:
    """
    Cheap pre-screen of plausible rotation matrices R using only signed permutations
    on the FFT magnitude (translation-invariant). Returns a list of 3x3 integer
    orthogonal matrices (± permutation matrices).
    Notes:
      - Effective & safe for cubic/tetragonal/orthorhombic where many sym ops are signed perms.
      - For hexagonal/trigonal/monoclinic/triclinic we will *not* use this to filter Halls.
    """
    Ss = signed_permutation_matrices()
    corrs = []
    for S in Ss:
        B = _permute_fft_by_signed_perm(Fmag, S)
        corrs.append(_cosine_corr(Fmag, B))
    corrs = np.array(corrs, float)
    # pick top-k with a small floor to ensure some diversity
    order = np.argsort(-corrs)  # descending
    k = max(int(min(len(Ss), topk)), int(min_keep))
    idx = order[:k].tolist()
    return [np.array(Ss[i], int) for i in idx]

def select_sg_by_hall(
    rho: np.ndarray,
    rho_coarse: np.ndarray,
    crystal_system: str,
    n_samples: int = 20000,
    seed: int = 0,
    compute_amp_corr: bool = False,
    limit_halls_per_sg: Optional[int] = None,
    P_family: str = "bravais",            # Default to bravais, covering F/I, etc.
    workers: int = 1,
    bravais_permute_axes: bool = True,
    centering: Optional[str] = None,
    fft_prescreen: bool = True,
    fft_topk: int = 24,
    fft_overlap_thresh: float = 0.20,
    score_on_coarse: Optional[bool] = None,   # NEW: control whether scoring uses coarse grid (True) or original grid (False/None)
) -> Tuple[HallScore, List[HallScore]]:
    lo, hi = _sg_range(crystal_system)
    if P_family == "signedperm":
        Ps = [M.astype(float) for M in signed_permutation_matrices()]
    elif P_family == "identity":
        Ps = [np.eye(3, dtype=float)]
    elif P_family == "bravais":
        Ps = standard_fractional_conjugations(crystal_system, permute_axes=bravais_permute_axes)
    else:
        raise ValueError("Unsupported P_family")

    # ----- FFT prescreen of R (only for cubic/tetragonal/orthorhombic) -----
    apply_fft_filter = (crystal_system.lower() in {"cubic","tetragonal","orthorhombic"}) and bool(fft_prescreen)
    Fmag_local = _fft_magnitude_from_rho_coarse(rho_coarse) if apply_fft_filter else None
    R_fft_cands = screen_Rs_by_fft(Fmag_local, crystal_system, topk=int(fft_topk)) if apply_fft_filter else []
    R_fft_set = {tuple(np.array(R).reshape(-1).tolist()) for R in R_fft_cands}

    # NEW: choose which grid to use for scoring according to the switch
    rho_for_score = (rho_coarse if score_on_coarse else rho)

    tasks = []
    for n in range(lo, hi+1):
        infos = _hall_infos_for_sg_number(n)
        # First filter by centering (P/A/B/C/F/I/R)
        if centering is not None:
            cent = centering.upper()
            infos = [h for h in infos if h["sg_symbol"] and str(h["sg_symbol"])[0].upper() == cent]
        # Then limit how many Hall settings per SG
        if limit_halls_per_sg is not None:
            infos = infos[:int(limit_halls_per_sg)]
        for info in infos:
            hall = int(info["hall_number"])
            # optional FFT-based filtering of this Hall for each P
            for P in Ps:
                if apply_fft_filter:
                    symdb_tmp = spg.get_symmetry_from_database(hall)
                    Rm_tmp = np.array(symdb_tmp["rotations"] if isinstance(symdb_tmp, dict) else symdb_tmp.rotations, dtype=int)
                    P_use = np.array(P, float)
                    R_use = (P_use @ Rm_tmp @ np.linalg.inv(P_use)).round().astype(int)
                    # Compute overlap ratio with FFT-screened signed-permutation set
                    Rset = {tuple(r.reshape(-1).tolist()) for r in R_use}
                    overlap = len(Rset & R_fft_set) / max(1, len(Rset))
                    if overlap < float(fft_overlap_thresh):
                        continue  # skip this Hall under this P
                tasks.append((hall, P, n_samples, seed, compute_amp_corr))

    #print("rho_for_score shape:", rho_for_score.shape)
    #print("lo, hi+1: ", lo, hi+1)
    #print("centering:", centering)
    #print("len(infos):", len(infos))
    #print("len(tasks)", len(tasks))
    #print("tasks: ", tasks)
    A_fft = fftn(rho_coarse)

    if workers > 1:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(rho_for_score, rho_coarse, A_fft, n_samples, seed, compute_amp_corr, (_fft_magnitude_from_rho_coarse(rho_coarse) if apply_fft_filter else None)),
        ) as pool:
            scores = list(pool.map(_score_task, tasks))
    else:
        _init_worker(rho_for_score, rho_coarse, A_fft, n_samples, seed, compute_amp_corr, (_fft_magnitude_from_rho_coarse(rho_coarse) if 'apply_fft_filter' in locals() and apply_fft_filter else None))
        scores = [_score_task(t) for t in tasks]

    #print("scores:", scores)

    scores = [s for s in scores if s is not None]
    if len(scores) == 0:
        return None, None
    scores.sort(key=lambda x: (x.avg_rmse, x.median_rmse))
    return scores[0], scores

# =================== report ===================
@dataclass
class SGReport:
    file: str
    grid_shape: Tuple[int,int,int]
    lattice_matrix: List[List[float]]
    crystal_system_hint: str
    best_sg_number: int
    best_sg_symbol: str
    best_hall_number: int
    best_avg_rmse: float
    best_median_rmse: float
    origin_shift_s: List[float]
    conjugation_P: List[List[float]]
    operations: List[Dict]
    top_alternatives: List[Dict]
    notes: str

def analyze_chgcar_to_sg(
    chgcar_path: str,
    crystal_system: str,
    n_samples: int = 25000,
    coarse_maxdim: int = 64,
    seed: int = 0,
    compute_amp_corr: bool = False,
    limit_halls_per_sg: Optional[int] = None,
    P_family: str = "bravais",
    workers: int = 1,
    bravais_permute_axes: bool = True,
    centering: Optional[str] = None,
    fft_prescreen: bool = True,
    fft_topk: int = 24,
    fft_overlap_thresh: float = 0.20,
) -> SGReport:
    """End-to-end: CHGCAR -> (best Hall setting), with optional auto-centering prepass."""
    g = load_chgcar_total(chgcar_path)
    rho = g.rho; lattice = g.lattice
    rho_coarse = downsample_wrap_to_maxdim(rho, coarse_maxdim, dtype=np.float32)

    tried_centerings: List[str] = []
    center_scores: Optional[Dict[str, float]] = None
    chosen_centering: Optional[str] = centering

    # NEW: global candidate pool across centerings (does not modify HallScore; used for global ranking/dedup later)
    all_scores_global: List[Tuple[HallScore, str]] = []

    if centering is None:
        # Auto pre-estimate candidates (typically 1–2 labels)
        tried_centerings, center_scores = estimate_centering(
            rho, crystal_system,
            n_samples=max(2000, n_samples // 6),
            coarse_maxdim=min(coarse_maxdim, 48),
            seed=seed,)
        if not tried_centerings:
            tried_centerings = ["P"]

        # Ensure at least P + “best non-P”
        if center_scores:
            order = sorted(center_scores.items(), key=lambda kv: kv[1])
            best_nonP = [lab for lab, _ in order if lab != "P"]
            if best_nonP:
                tried_centerings = list(dict.fromkeys(["P", best_nonP[0]] + tried_centerings))
            else:
                tried_centerings = list(dict.fromkeys(["P"] + tried_centerings))
        else:
            tried_centerings = list(dict.fromkeys(["P"] + tried_centerings))

        # Evaluate each centering branch separately, pick branch winner by rmse
        best: Optional[HallScore] = None
        all_scores: Optional[List[HallScore]] = None
        chosen_centering = None

        for c in tried_centerings:
            b, scores_c = select_sg_by_hall(
                rho, rho_coarse,
                crystal_system=crystal_system,
                n_samples=n_samples, seed=seed,
                compute_amp_corr=compute_amp_corr,
                limit_halls_per_sg=limit_halls_per_sg, P_family=P_family,
                workers=workers, bravais_permute_axes=bravais_permute_axes, centering=c,
            )
            # NEW: merge candidates (tag with centering for later global ranking)
            all_scores_global.extend((s, c) for s in scores_c)

            if (best is None) or (b.avg_rmse < best.avg_rmse) or (b.avg_rmse == best.avg_rmse and b.median_rmse < best.median_rmse):
                best, all_scores, chosen_centering = b, scores_c, c
    else:
        # Centering given explicitly
        tried_centerings = [centering]
        best, all_scores = select_sg_by_hall(
            rho, rho_coarse,
            crystal_system=crystal_system,
            n_samples=n_samples, seed=seed,
            compute_amp_corr=compute_amp_corr,
            limit_halls_per_sg=limit_halls_per_sg, P_family=P_family,
            workers=workers, bravais_permute_axes=bravais_permute_axes, centering=centering,
        )
        # NEW: when centering is explicit, still add this branch's candidates to the global pool
        all_scores_global.extend((s, centering) for s in all_scores)

    # pack outputs (keep your original output structure)
    ops_out = [{
        "R": np.array(op.R, int).tolist(),
        "t": [float(x) for x in wrap01(op.t_eff)],
        "rmse": float(op.rmse),
        "variant": op.variant,
        "delta_axis": float(op.delta_axis),
        "bravais_shift": [float(x) for x in (op.bravais_shift if isinstance(op.bravais_shift, (list, tuple, np.ndarray)) else [0.0, 0.0, 0.0])],
        "amp_corr": (None if op.amp_corr is None else float(op.amp_corr)),
    } for op in best.op_scores]

    # NEW: do global ranking & dedup across centerings (dedup by (sg_number, hall_number))
    all_scores_global_sorted = sorted(all_scores_global, key=lambda sc: (sc[0].avg_rmse, sc[0].median_rmse))
    seen = set()
    alt = []
    for (s, c) in all_scores_global_sorted:
        key = (int(s.sg_number), int(s.hall_number))
        if key in seen:
            continue
        seen.add(key)
        alt.append({
            "centering": c,
            "sg_number": int(s.sg_number),
            "sg_symbol": str(s.sg_symbol),
            "hall_number": int(s.hall_number),
            "avg_rmse": float(s.avg_rmse),
            "median_rmse": float(s.median_rmse),
        })
        if len(alt) >= 15:
            break

    # Notes on centering path (keep your original explanatory logic)
    if center_scores is not None:
        score_str = ", ".join(f"{k}:{center_scores[k]:.4f}" for k in sorted(center_scores))
        note_extra = f" Auto-centering: tried={tried_centerings}, scores={{" + score_str + f"}}, chosen={chosen_centering}."
    else:
        note_extra = f" Centering fixed to {tried_centerings[0]}." if tried_centerings else ""

    return SGReport(
        file=os.path.abspath(chgcar_path),
        grid_shape=tuple(int(x) for x in rho.shape),
        lattice_matrix=lattice.matrix.tolist(),
        crystal_system_hint=crystal_system,
        best_sg_number=int(best.sg_number),
        best_sg_symbol=str(best.sg_symbol),
        best_hall_number=int(best.hall_number),
        best_avg_rmse=float(best.avg_rmse),
        best_median_rmse=float(best.median_rmse),
        origin_shift_s=[float(x) for x in best.origin_shift_s],
        conjugation_P=np.array(best.P_conjugation, float).tolist(),
        operations=ops_out,
        top_alternatives=alt,
        notes=("RMSE uses per-op Bravais half-lattice and C2-axis compensation with per-op {R,RT} choice; "
               "group score = trimmed median with log-symmetry bonus." + note_extra),
    )

def save_report_json(report: SGReport, out_path: Optional[str] = None) -> str:
    if out_path is None:
        base = os.path.basename(report.file)
        out_path = os.path.join(os.path.dirname(report.file), base + ".rho_sg.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)
    return out_path

# =================== probe (optional) ===================
def probe_sg_once(chgcar_path: str, sg_number: int = 227, hall_number: int | None = None,
                  system: str = "cubic", coarse_maxdim: int = 48, seed: int = 0):
    g = load_chgcar_total(chgcar_path)
    rho = g.rho
    rho_c = downsample_wrap_to_maxdim(rho, coarse_maxdim, dtype=np.float32)
    A_fft = fftn(rho_c)
    coords = _precompute_frac_grid(rho_c.shape)
    # ... keep the original probing logic ...

def compare_sg_from_report(report, structure=None, chgcar_path=None, symprec: float = 1e-3, angle_tolerance: float = 0.5):
    """
    Compare pymatgen's space group from the lattice (SG_lattice) with an already-computed
    report's rho-based space group (SG_rho) without recomputing ρ→SG.
    Prefer the provided `structure`; otherwise try to read from `chgcar_path`;
    otherwise fall back to `report.file` / `report.path`. Returns a lightweight dict.
    """
    from pymatgen.io.vasp.outputs import Chgcar
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    def _crystal_system_from_sgnum(n: int) -> str:
        if 1 <= n <= 2: return "triclinic"
        if 3 <= n <= 15: return "monoclinic"
        if 16 <= n <= 74: return "orthorhombic"
        if 75 <= n <= 142: return "tetragonal"
        if 143 <= n <= 167: return "trigonal"
        if 168 <= n <= 194: return "hexagonal"
        if 195 <= n <= 230: return "cubic"
        return "unknown"

    # --- SG_rho comes from the precomputed report ---
    rho_num = int(getattr(report, "best_sg_number"))
    rho_sym = str(getattr(report, "best_sg_symbol"))
    rho_sys = _crystal_system_from_sgnum(rho_num)
    rho_cent = rho_sym.strip()[0].upper() if rho_sym else "?"

    # --- Prepare structure (used only for pymatgen analysis) ---
    if structure is None:
        if chgcar_path is None:
            chgcar_path = getattr(report, "file", None) or getattr(report, "path", None)
        if chgcar_path is None:
            raise ValueError("需要提供 structure，或 chgcar_path，或在 report 中包含 file/path。")
        structure = Chgcar.from_file(chgcar_path).structure

    # --- SG_lattice by pymatgen ---
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    pmg_num = int(sga.get_space_group_number())
    pmg_sym = str(sga.get_space_group_symbol())
    pmg_sys = str(sga.get_lattice_type())  # 'cubic','tetragonal','orthorhombic',...
    pmg_cent = pmg_sym.strip()[0].upper() if pmg_sym else "?"

    # --- Comparison ---
    bravais_match = (rho_sys.lower() == pmg_sys.lower()) and (rho_cent == pmg_cent)
    sg_match = (rho_num == pmg_num)

    return {
        "file": chgcar_path or getattr(report, "file", None) or getattr(report, "path", None),
        "bravais_match": bool(bravais_match),
        "sg_match": bool(sg_match),
        "rho": { "system": rho_sys, "centering": rho_cent, "number": rho_num, "symbol": rho_sym },
        "pmg": { "system": pmg_sys, "centering": pmg_cent, "number": pmg_num, "symbol": pmg_sym },
        "notes": "仅复用 report 的 ρ→SG 结果；结构侧用 pymatgen 做一次 SG 分析进行对比。",
    }

# =================== main ===================
def main():
    global DEBUG, DEBUG_HALLS, DEBUG_PRINT_LIMIT, DEBUG_RESID_CSV, DEBUG_SAMPLING_OVERRIDE
    global TRIM_FRAC, SYM_BONUS_LOG

    ap = argparse.ArgumentParser(description="ρ→SG (robust & fair: Bravais/C2 compensation + trimmed-median + symmetry bonus)")
    ap.add_argument("inputs", nargs="+", help="CHGCAR paths (files or globs)")
    ap.add_argument("--system", required=True, type=str,
                    choices=["triclinic","monoclinic","orthorhombic","tetragonal","trigonal","hexagonal","cubic"])
    ap.add_argument("--nsamples", type=int, default=25000)
    ap.add_argument("--coarse", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ampcorr", action="store_true")
    ap.add_argument("--limit-halls-per-sg", type=int, default=4)
    ap.add_argument("--P-family", type=str, default="bravais",
                    choices=["signedperm","identity","bravais"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-perm", action="store_true", default=True,
                    help="When --P-family bravais: don't left-compose with permutations")
    ap.add_argument("--centering", type=str, default=None, choices=list("PABCIFR"))
    ap.add_argument("--summary", type=str, default=None)
    ap.add_argument("--fast", action="store_true", default=True)
    # FFT prescreen controls
    ap.add_argument("--no-fft-prescreen", action="store_true",
                    help="Disable FFT magnitude-based R pre-screen (default on for cubic/tetragonal/orthorhombic)")
    ap.add_argument("--fft-topk", type=int, default=24,
                    help="How many signed-permutation R candidates to keep from FFT screening")
    ap.add_argument("--fft-overlap", type=float, default=0.20,
                    help="Minimum overlap ratio between a Hall's R set and FFT-screened R candidates to evaluate that Hall")

    # Debug
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-hall", type=int, action="append", default=[])
    ap.add_argument("--debug-rt", type=str, default="Rt", choices=["Rt","R"])
    ap.add_argument("--debug-print", type=int, default=10)
    ap.add_argument("--debug-resid-csv", type=str, default=None)

    # Expose tunables
    ap.add_argument("--trim-frac", type=float, default=TRIM_FRAC)
    ap.add_argument("--sym-bonus-log", type=float, default=SYM_BONUS_LOG)

    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-sg", type=int, default=None)
    ap.add_argument("--probe-hall", type=int, default=None)

    args = ap.parse_args()

    DEBUG = bool(args.debug)
    DEBUG_HALLS = set(args.debug_hall or [])
    DEBUG_PRINT_LIMIT = int(args.debug_print)
    if args.debug_resid_csv:
        DEBUG_RESID_CSV = args.debug_resid_csv
    if args.debug_rt:
        DEBUG_SAMPLING_OVERRIDE = (args.debug_rt == "Rt")
    TRIM_FRAC = float(args.trim_frac)
    SYM_BONUS_LOG = float(args.sym_bonus_log)

    reports: List[SGReport] = []
    paths: List[str] = []
    for inp in args.inputs:
        if any(ch in inp for ch in "*?[]"):
            paths.extend(glob(inp))
        else:
            paths.append(inp)

    for path in paths:
        rep = analyze_chgcar_to_sg(
            path,
            crystal_system=args.system,
            n_samples=args.nsamples,
            coarse_maxdim=args.coarse,
            seed=args.seed,
            compute_amp_corr=args.ampcorr,
            limit_halls_per_sg=args.limit_halls_per_sg,
            P_family=args.P_family,
            workers=args.workers,
            bravais_permute_axes=(not args.no_perm),
            centering=args.centering,
        )
        out_json = save_report_json(rep)
        print(f"ok: best={rep.best_sg_number} {rep.best_sg_symbol} hall={rep.best_hall_number} "
              f"avg_rmse={rep.best_avg_rmse:.6f} out={out_json}")
        reports.append(rep)

    if args.summary and reports:
        agg = {"files":[r.file for r in reports],
               "best":[int(r.best_sg_number) for r in reports],
               "best_symbol":[r.best_sg_symbol for r in reports],
               "avg_rmse":[float(r.best_avg_rmse) for r in reports]}
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    res = compare_sg_from_report(rep)  # If rep.file/rep.path points to CHGCAR, it will be read automatically
    print(res["bravais_match"], res["sg_match"])

if __name__ == "__main__":
    main()

