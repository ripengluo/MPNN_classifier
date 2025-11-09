#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From CHGCAR (valence rho) to Wyckoff occupancy
"""

from __future__ import annotations
import sys, json, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from numpy.linalg import eigh, norm, inv, eigvalsh
from scipy import ndimage as ndi
from cnn import *
from pyxtal.symmetry import Group

import spglib
from pymatgen.io.vasp.outputs import Chgcar
from skimage.feature import peak_local_max as _skimage_peak_local_max

HAS_PMG = True
SKIMAGE_AVAILABLE = True

import re
from fractions import Fraction


# ---- data structures ----
@dataclass
class AtomicPeak:
    frac: np.ndarray
    cart: np.ndarray
    shape: str
    sphericity: float
    elongation: float
    peak_value: float
    est_radius: float
    est_charge: float

# ---- I/O ----
def read_chgcar(path: str):
    if not HAS_PMG:
        print("[ERROR] pymatgen is required: pip install pymatgen", file=sys.stderr)
        sys.exit(2)
    chg = Chgcar.from_file(path)
    grid = chg.data["total"]  # (nx,ny,nz)
    lattice = chg.structure.lattice.matrix  # 3x3 Å
    nx, ny, nz = grid.shape
    vol = float(np.linalg.det(lattice))
    dv = vol/(nx*ny*nz)
    tot_e = float(np.sum(grid) * dv)
    return type("CHG", (), dict(grid=grid, lattice=lattice, volume=vol, dv=dv, total_electrons=tot_e))

# ---- utils ----
def _string_to_fracs(s: str) -> List[Tuple[str,str,str]]:
    TOKEN = r'[+-]?(?:\d+(?:/\d+)?|[A-Za-z]\w*)'
    LINE_RE = rf'^\s*({TOKEN})\s*,\s*({TOKEN})\s*,\s*({TOKEN})\s*$'
    triplets = re.findall(LINE_RE, s, flags=re.M)
    return triplets  # coords (only lines where each component is a single token will be captured)

def wrap_frac(frac: np.ndarray) -> np.ndarray:
    f = np.array(frac, dtype=float)
    return f - np.floor(f)

def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return (wrap_frac(frac)) @ lattice

def cart_to_frac(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return np.asarray(cart, float) @ inv(lattice)

def minimum_image_cart(v: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    f = cart_to_frac(v, lattice); f -= np.round(f)
    return frac_to_cart(f, lattice)

def trilinear_sample(grid: np.ndarray, frac: np.ndarray) -> float:
    """
    Sample using i/N convention (no -0.5). Coordinates in [0, N) with wrap.
    """
    nx, ny, nz = grid.shape
    coords = np.array([[frac[0]*nx, frac[1]*ny, frac[2]*nz]]).T
    val = ndi.map_coordinates(grid, coords, order=1, mode='wrap')
    return float(val.reshape(-1)[0])

def subvoxel_refine(grid: np.ndarray, idx):
    """
    3-point quadratic fit per axis (PBC); i/N → frac in [0,1)
    """
    nx, ny, nz = grid.shape
    ix, iy, iz = [int(x) for x in idx]

    def quad_peak(fm, f0, fp):
        denom = (fm - 2.0*f0 + fp)
        if abs(denom) < 1e-12:
            return 0.0
        delta = 0.5 * (fm - fp) / denom
        return float(np.clip(delta, -0.5, 0.5))

    xm = grid[(ix-1) % nx, iy, iz]; x0 = grid[ix % nx, iy, iz]; xp = grid[(ix+1) % nx, iy, iz]
    ym = grid[ix, (iy-1) % ny, iz]; y0 = grid[ix, iy % ny, iz]; yp = grid[ix, (iy+1) % ny, iz]
    zm = grid[ix, iy, (iz-1) % nz]; z0 = grid[ix, iy, iz % nz]; zp = grid[ix, iy, (iz+1) % nz]

    dx = quad_peak(xm, x0, xp)
    dy = quad_peak(ym, y0, yp)
    dz = quad_peak(zm, z0, zp)

    x_cont = (ix + dx) % nx
    y_cont = (iy + dy) % ny
    z_cont = (iz + dz) % nz

    frac = np.array([x_cont / nx, y_cont / ny, z_cont / nz], dtype=float)
    val = ndi.map_coordinates(grid, np.array([[x_cont, y_cont, z_cont]]).T, order=1, mode='wrap')
    return frac, float(val.reshape(-1)[0])

def _circular_mean_frac(frac_arr: np.ndarray, weights: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Circular mean along each fractional axis; result ∈ [0,1).
    """
    fr = wrap_frac(np.asarray(frac_arr, float))
    if fr.size == 0:
        return np.zeros(3, float)
    if weights is None:
        weights = np.ones(len(fr), float)
    w = np.asarray(weights, float).reshape(-1)
    ang = 2.0*np.pi*fr
    s = np.sum(np.sin(ang) * w[:, None], axis=0)
    c = np.sum(np.cos(ang) * w[:, None], axis=0)
    theta = np.arctan2(s, c)
    return wrap_frac(theta/(2.0*np.pi))

# ---- peak finder wrapper (skimage compatibility) ----
if SKIMAGE_AVAILABLE:
    # Omit "num_peaks" when None to avoid np.isfinite(None) errors in newer scikit-image
    def peak_local_max(g, min_distance_vox=3, threshold_rel=0.1,
                       exclude_border=False, num_peaks=None, footprint=None, **kwargs):
        kw = dict(
            min_distance=int(min_distance_vox),
            threshold_rel=float(threshold_rel),
            exclude_border=exclude_border,
            footprint=footprint,
        )
        if num_peaks is not None:
            kw["num_peaks"] = int(num_peaks)
        return _skimage_peak_local_max(g, **kw)
else:
    def peak_local_max(g, min_distance_vox=3, threshold_rel=0.1,
                       exclude_border=False, num_peaks=None, footprint=None, **kwargs):
        if footprint is None:
            footprint = np.ones((3,3,3), dtype=bool)
        thr = float(g.min()) + float(threshold_rel)*(float(g.max())-float(g.min()))
        local_max = (g >= thr) & (g == ndi.maximum_filter(g, footprint=footprint, mode='wrap'))
        coords = np.array(np.nonzero(local_max)).T
        if coords.size == 0: return coords
        intens = g[tuple(coords.T)]
        order = np.argsort(-intens); coords = coords[order]
        sel = []
        shape = np.array(g.shape)
        def far_enough(c, chosen):
            if not chosen: return True
            d = np.abs(np.array(chosen) - c)
            d = np.minimum(d, shape - d)
            dist = np.sqrt((d**2).sum(axis=1))
            return np.all(dist >= float(min_distance_vox))
        for c in coords:
            if far_enough(c, sel):
                sel.append(tuple(c))
            if (num_peaks is not None) and (len(sel) >= int(num_peaks)):
                break
        return np.array(sel, dtype=int)

def find_peaks(grid, min_distance_vox=3, threshold_rel=0.1,
               exclude_border=False, num_peaks=None, footprint=None, mode='max'):
    """
    Relative-threshold peak finding (simple adaptive baseline).
    - threshold_rel: single criterion in [0,1], applied relative to [min(grid), max(grid)].
    - mode: 'max' (use grid) or 'min' (use -grid).
    """
    g = grid if mode == 'max' else -grid
    return peak_local_max(
        g,
        min_distance_vox=int(min_distance_vox),
        threshold_rel=float(threshold_rel),
        exclude_border=exclude_border,
        num_peaks=(None if num_peaks is None else int(num_peaks)),
        footprint=(np.ones((3,3,3), dtype=bool) if footprint is None else footprint)
    )

# ---- simple top-fraction clustering (PBC, Å-radius) ----
def _build_neighbor_offsets(nx: int, ny: int, nz: int, lattice: np.ndarray, R_A: float) -> np.ndarray:
    """
    Precompute integer voxel offsets (dx,dy,dz) whose real-space distance ≤ R_A (Å).
    Use an average voxel size to conservatively bound the voxel window to reduce scanning.
    """
    lattice = np.asarray(lattice, float)
    aA, bA, cA = norm(lattice[0]), norm(lattice[1]), norm(lattice[2])
    vox_A = (aA/nx + bA/ny + cA/nz) / 3.0
    r_vox = int(np.ceil(max(R_A / max(vox_A, 1e-12), 1.0)))
    offsets = []
    for dx in range(-r_vox, r_vox+1):
        for dy in range(-r_vox, r_vox+1):
            for dz in range(-r_vox, r_vox+1):
                df = np.array([dx/nx, dy/ny, dz/nz], float)
                dA = float(np.linalg.norm(df @ lattice))
                if dA <= (R_A + 1e-12):
                    offsets.append((dx,dy,dz))
    if (0,0,0) not in offsets:
        offsets.append((0,0,0))
    return np.asarray(offsets, int)


# ---- merging: NMS + PBC circular mean ----
def merge_close_peaks(fracs, pvals, lattice, merge_radius_cart,
                      grid, **kwargs):
    """
    Greedy NMS with PBC: strongest-first assignment + circular mean per cluster.
    """
    fracs = np.asarray(fracs, float).reshape(-1,3)
    pvals = np.asarray(pvals, float).reshape(-1)
    if fracs.size == 0:
        return fracs, pvals

    lattice = np.asarray(lattice, float)

    def wrap(f): return np.mod(f, 1.0)
    def min_img_df(df): return df - np.round(df)
    def distA(f1, f2): return float(np.linalg.norm((min_img_df(f1 - f2) @ lattice)))

    order = np.argsort(-pvals)
    clusters = []
    centers_frac = []

    def circular_mean(fr_arr, w):
        w = np.asarray(w, float).reshape(-1)
        if fr_arr.size == 0: return np.zeros(3)
        ang = 2.0*np.pi*fr_arr
        s = np.sum((np.sin(ang).T * w).T, axis=0)
        c = np.sum((np.cos(ang).T * w).T, axis=0)
        theta = np.arctan2(s, c)
        fr = (theta/(2.0*np.pi)) % 1.0
        return fr

    R = float(merge_radius_cart)
    for idx in order:
        f = fracs[idx]
        assigned = False
        if centers_frac:
            dists = np.array([distA(f, cf) for cf in centers_frac])
            j = int(np.argmin(dists))
            if dists[j] <= R:
                clusters[j].append(idx)
                frs = fracs[clusters[j]]
                ws  = pvals[clusters[j]]
                centers_frac[j] = circular_mean(frs, ws)
                assigned = True
        if not assigned:
            clusters.append([idx])
            centers_frac.append(f.copy())

    out_f, out_v = [], []
    for ids, cf in zip(clusters, centers_frac):
        pv = float(np.max(pvals[ids]))
        out_f.append(wrap(cf))
        out_v.append(pv)

    out_f = np.asarray(out_f); out_v = np.asarray(out_v)
    order2 = np.argsort(-out_v)
    return out_f[order2], out_v[order2]

# ---- local shape metrics (90%) ----
def local_shape_metrics(
    grid: np.ndarray,
    frac: np.ndarray,
    lattice: np.ndarray,
    level_rel: float = 0.95,
    radius_vox: int = 3,
    keep_ratio: float = 0.90,
    pca_radius_A: float = 1.6,
    neigh_R_A: float = 1.5,
    s_max_A: float = 2.6,
    ds_A: float = 0.02
):
    G = np.asarray(grid, float)
    lattice = np.asarray(lattice, float)
    frac = wrap_frac(np.asarray(frac, float))
    nx, ny, nz = G.shape
    LinvT = inv(lattice).T
    eps = 1e-16

    def f2c(f): return (wrap_frac(f)) @ lattice

    c0 = f2c(frac); rho0 = trilinear_sample(G, frac)

    # PCA neighborhood
    a = norm(lattice[0]); b = norm(lattice[1]); c = norm(lattice[2])
    vox = (a/nx + b/ny + c/nz)/3.0
    step = max(vox/1.2, 0.05)
    r = float(pca_radius_A)
    grid_lin = np.arange(-r, r+1e-12, step)
    X, Y, Z = np.meshgrid(grid_lin, grid_lin, grid_lin, indexing='ij')
    mask = (X*X + Y*Y + Z*Z) <= (r*r + 1e-12)
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
    if pts.size == 0:
        return "other", 0.0, 1.0, 0.0, np.array([1.0,0.0,0.0])

    fr_pts = ((c0[None,:] + pts) @ LinvT) % 1.0
    coords = (fr_pts*np.array([nx,ny,nz], float))
    rhos = ndi.map_coordinates(G, coords.T, order=1, mode='wrap')
    w = rhos - rhos.min()
    if np.sum(w) <= 0: w = np.ones_like(rhos)
    mu = (pts * w[:,None]).sum(axis=0)/np.sum(w)
    Xc = pts - mu
    C = (Xc.T * w) @ Xc / max(float(np.sum(w)), 1e-16)
    vals, vecs = eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    lam1, lam2, lam3 = (vals.tolist() + [0.0, 0.0, 0.0])[:3]
    v1 = vecs[:,0] / (norm(vecs[:,0]) + eps)
    v2 = vecs[:,1] / (norm(vecs[:,1]) + eps)
    v3 = np.cross(v1, v2); v3 /= (norm(v3) + eps)
    sphericity = (lam3 / (lam1 + eps)) if lam1 > eps else 0.0

    # baseline and 1D lengths
    idx_R = max(1, int(np.ceil(neigh_R_A / max(vox, 1e-12))))
    rng = np.arange(-idx_R, idx_R+1)
    DX, DY, DZ = np.meshgrid(rng, rng, rng, indexing='ij')
    offs = np.stack([DX.ravel()/nx, DY.ravel()/ny, DZ.ravel()/nz], axis=1)
    fn = (frac[None,:] + offs) % 1.0
    cn = fn @ lattice
    maskn = np.linalg.norm(cn - c0[None,:], axis=1) <= (neigh_R_A+1e-12)
    fn = fn[maskn]
    coords = (fn*np.array([nx,ny,nz], float))
    neigh_vals = ndi.map_coordinates(G, coords.T, order=1, mode='wrap')
    rho_min = float(np.min(neigh_vals)) if neigh_vals.size else rho0
    thr = rho_min + keep_ratio*(rho0 - rho_min)

    s = np.arange(-float(s_max_A), float(s_max_A)+1e-12, float(ds_A))
    def length_along(u):
        u = u/(norm(u)+eps)
        fr_line = ((c0[None,:] + s[:,None]*u[None,:]) @ LinvT) % 1.0
        coords = (fr_line*np.array([nx,ny,nz], float))
        rho_line = ndi.map_coordinates(G, coords.T, order=1, mode='wrap')
        pos = np.where(s>=0)[0]; neg = np.where(s<=0)[0]
        def endpoint(idx, positive=True):
            rr = rho_line[idx]; ss = s[idx]
            hit = np.where(rr <= thr)[0]
            if hit.size == 0:
                return ss[-1] if positive else ss[0]
            k = hit[0] if positive else hit[-1]
            if positive:
                if k == 0: return ss[0]
                s1, s2 = ss[k-1], ss[k]; r1, r2 = rr[k-1], rr[k]
            else:
                if k == len(ss)-1: return ss[-1]
                s1, s2 = ss[k], ss[k+1]; r1, r2 = rr[k], rr[k+1]
            t = 0.0 if (r2 == r1) else (thr - r1)/(r2 - r1)
            return float(s1 + t*(s2 - s1))
        sp = endpoint(pos, True); sm = endpoint(neg, False)
        return float((sp - 0.0) + (0.0 - sm))

    L_par   = length_along(v1)
    W_perp1 = length_along(v2)
    W_perp2 = length_along(v3)
    mean_w  = (W_perp1 + W_perp2)/2.0

    elongation = L_par/max(mean_w, 1e-12)
    est_radius = L_par*0.5

    if sphericity >= 0.90 and elongation < 1.4:
        shape = "sphere"
    elif (elongation >= 1.70) and (sphericity <= 0.92):
        shape = "rod"
    else:
        # Slightly relax for metals/ions
        shape = "sphere" if (sphericity >= 0.80 or elongation < 1.25) else "other"

    return shape, float(sphericity), float(elongation), float(est_radius), v1

# ---- Density & PeakDetector ----
class DensityField:
    def __init__(self, chg, smooth_sigma_vox=1.0):
        self.grid = chg.grid
        self.lattice = chg.lattice
        self.dv = chg.dv
        self.volume = chg.volume
        self.total_electrons = chg.total_electrons
        # One smoothing for stability
        self.grid = ndi.gaussian_filter(self.grid, sigma=float(smooth_sigma_vox))
    @property
    def shape(self): return self.grid.shape

def detect_peaks_msblob(grid, lattice,
                        sigmas_A=(0.40, 0.55, 0.70, 0.90, 1.20),
                        z_mad=3.0, nms_sep_A=0.80):
    """
    Multi-scale LoG + MAD adaptive threshold + PBC merging.
    Returns: (fracs[K,3], peak_values[K])
    """
    import numpy as np
    from numpy.linalg import norm
    from scipy import ndimage as ndi

    G = np.asarray(grid, float)
    nx, ny, nz = G.shape
    aA, bA, cA = float(norm(lattice[0])), float(norm(lattice[1])), float(norm(lattice[2]))
    sx, sy, sz = aA/nx, bA/ny, cA/nz  # voxel sizes (Å) along three axes

    # 1) Multi-scale LoG (anisotropic σ converted Å→voxel)
    R_all = []
    for sA in sigmas_A:
        sig_vox = (max(sA/sx, 0.5), max(sA/sy, 0.5), max(sA/sz, 0.5))  # lower bound 0.5 voxel to avoid numerical noise
        R = ndi.gaussian_laplace(G, sigma=sig_vox)
        R_all.append((sA**2) * R)  # scale normalization
    R_stack = np.stack(R_all, axis=0)       # (S,nx,ny,nz)
    Rmin   = R_stack.min(axis=0)            # take strongest "blob" response (more negative = peak-like)
    # 2) MAD threshold (only on negatives)
    neg = -Rmin[Rmin < 0]
    if neg.size == 0:
        return np.zeros((0,3)), np.zeros((0,))
    med = np.median(neg)
    mad = np.median(np.abs(neg - med)) + 1e-12
    tau = med + z_mad * mad
    good = (-Rmin) >= tau

    # 3) Do local maxima on -Rmin + PBC merge
    #    min_distance estimated by Å→voxel; reuse your existing find_peaks/NMS wrapper
    voxel_A = (sx + sy + sz) / 3.0
    min_dist_vox = max(2, int(round(float(nms_sep_A) / max(voxel_A, 1e-9))))
    coords = find_peaks(-Rmin, min_distance_vox=min_dist_vox,
                        threshold_rel=0.0, exclude_border=False, footprint=None, mode='max')
    coords = np.array([c for c in coords if good[tuple(c)]])
    fracs, pvals = [], []
    for (ix,iy,iz) in coords:
        f, val = subvoxel_refine(G, np.array([ix,iy,iz]))
        fracs.append(f); pvals.append(val)
    fracs = np.asarray(fracs); pvals = np.asarray(pvals)

    # PBC merge (Å scale)
    centers, groups = _cluster_points_frac(fracs, lattice, tol_A=float(nms_sep_A))
    out_f, out_v = [], []
    for g, cf in zip(groups, centers):
        best = g[int(np.argmax(pvals[g]))]
        out_f.append(cf); out_v.append(pvals[best])
    order = np.argsort(-np.asarray(out_v))
    return np.asarray(out_f)[order], np.asarray(out_v)[order]

def assign_wyckoff_for_msblob_peaks(
    grid: np.ndarray,
    lattice: np.ndarray,
    sg: int = 221,
    # LoG/NMS (keep consistent or slightly relaxed vs detect_peaks_msblob)
    sigmas_A=(0.10, 0.40, 0.70, 0.90, 1.20),
    z_mad: float = 3.0,
    nms_sep_A: float = 0.80,
    # atom/bond criterion (match shape criteria in file)
    sphere_sphericity_thr: float = 0.80,
    atom_elong_thr: float = 1.25,
    # BCP → atom params (reuse defaults from your existing function)
    bcp_keep_ratio: float = 0.90,
    pca_radius_A: float = 1.6,
    neigh_R_A: float = 1.5,
    s_max_A: float = 2.8,
    ds_A: float = 0.02,
    bcp_cluster_tol_A: float = 0.5,
    atom_merge_tol_A: float = 0.35,
    symprec: float = 1e-3,
):
    """
    Step 1: For peaks detected by detect_peaks_msblob, do atom/bond classification and Wyckoff labeling.
    - Atomic peaks: assign Wyckoff directly via spglib;
    - Bond peaks: back-calculate atoms using atomic_candidates_from_bcps, then assign Wyckoff;
    - Merge atoms from both paths and return the consolidated set and occupancy statistics.
    """
    # 1) Obvious peaks (multi-scale LoG + MAD + NMS)
    fr_raw, p_raw = detect_peaks_msblob(
        grid, lattice, sigmas_A=sigmas_A, z_mad=z_mad, nms_sep_A=nms_sep_A
    )

    # 2) Shape metrics (use local_shape_metrics / threshold logic already in this file)
    peaks = []
    atomic_like_fracs = []
    bond_like_bcp_fracs = []
    for f, val in zip(fr_raw, p_raw):
        shape, sph, elo, r_est, _ = local_shape_metrics(
            grid, f, lattice, level_rel=0.60, radius_vox=3
        )
        is_atomic = (sph >= sphere_sphericity_thr) or (elo < atom_elong_thr)
        peaks.append({
            "frac": wrap_frac(f).tolist(),
            "peak_value": float(val),
            "shape": shape,
            "sphericity": float(sph),
            "elongation": float(elo),
            "est_radius": float(r_est),
            "class": "atomic" if is_atomic else "bond"
        })
        (atomic_like_fracs if is_atomic else bond_like_bcp_fracs).append(wrap_frac(f))

    atomic_like_fracs = np.asarray(atomic_like_fracs, float) if atomic_like_fracs else np.zeros((0,3))
    bond_like_bcp_fracs = np.asarray(bond_like_bcp_fracs, float) if bond_like_bcp_fracs else np.zeros((0,3))

    # 3) Bond peaks → atoms (reuse your existing BCP-backtracking)
    atoms_from_bcps = np.zeros((0,3))
    if len(bond_like_bcp_fracs) > 0:
        atoms_from_bcps, _mults, _support = atomic_candidates_from_bcps(
            grid, lattice, bond_like_bcp_fracs,
            keep_ratio=bcp_keep_ratio, s_max_A=s_max_A, ds_A=ds_A,
            pca_radius_A=pca_radius_A, neigh_R_A=neigh_R_A,
            cluster_tol_A=bcp_cluster_tol_A, final_merge_tol_A=0.45
        )

    # 4) Merge two atom paths (PBC dedup)
    final_atoms = merge_atom_sets(atoms_from_bcps, atomic_like_fracs, lattice, tol_A=atom_merge_tol_A)

    # 5) Per-site Wyckoff (one-to-one with input order)
    import spglib
    def _per_site_wy(fracs):
        if fracs is None or len(fracs) == 0:
            return [], {}, None
        fr = wrap_frac(np.asarray(fracs, float))
        nums = np.ones(len(fr), dtype=int)
        ds = spglib.get_symmetry_dataset((np.asarray(lattice,float), fr, nums), symprec=float(symprec))
        wy = list(ds["wyckoffs"])
        occ = {}
        for w in wy: occ[w] = occ.get(w, 0) + 1
        return wy, occ, int(ds["number"])

    wy_atomic_list, wy_atomic_occ, sg_atomic = _per_site_wy(atomic_like_fracs)
    wy_bcps_list,   wy_bcps_occ,   sg_bcps   = _per_site_wy(atoms_from_bcps)
    wy_final_list,  wy_final_occ,  sg_final  = _per_site_wy(final_atoms)

    atoms_atomic = [
        {"frac": wrap_frac(f).tolist(), "wyckoff": w}
        for f, w in zip(atomic_like_fracs, wy_atomic_list)
    ] if len(atomic_like_fracs) else []

    atoms_from_bcps_list = [
        {"frac": wrap_frac(f).tolist(), "wyckoff": w}
        for f, w in zip(atoms_from_bcps, wy_bcps_list)
    ] if len(atoms_from_bcps) else []

    return {
        "peaks": peaks,
        "atoms_atomic": atoms_atomic,
        "atoms_from_bcps": atoms_from_bcps_list,
        "final_atoms": (wrap_frac(final_atoms).tolist() if len(final_atoms) else []),
        "final_wyckoff_occupancy": wy_final_occ,
        "spglib_sg_detected": {"atomic_path": sg_atomic, "bcp_path": sg_bcps, "final": sg_final},
        "sg_given": int(sg)
    }

class PeakDetector:
    """
    Minimal, msblob-only PeakDetector:
    - detect_peaks_msblob(...)  -> (fracs, peak_values)
    - classify_and_assign_wyckoff(sg=221, ...) -> directly delegates to assign_wyckoff_for_msblob_peaks
    Goal: avoid duplication with assign_wyckoff_for_msblob_peaks, keep single-responsibility and pluggability.
    """
    def __init__(self, F: DensityField):
        self.F = F  # F.grid (nx,ny,nz), F.lattice (3x3)

    # --- 1) msblob-only peak detection (thin wrapper) ---
    def detect_peaks_msblob(self,
                            sigmas_A=(0.10, 0.40, 0.70, 0.90, 1.20),
                            z_mad: float = 3.0,
                            nms_sep_A: float = 0.80,
                            peaks_as_atomicpeaks: bool = False):
        """
        Returns:
          - peaks_as_atomicpeaks=False: (fracs[K,3], peak_values[K])
          - peaks_as_atomicpeaks=True : List[AtomicPeak] (compute shape/coarse integral only if actually needed upstream)
        """
        fr_raw, p_raw = detect_peaks_msblob(
            self.F.grid, self.F.lattice,
            sigmas_A=sigmas_A, z_mad=z_mad, nms_sep_A=nms_sep_A
        )
        if not peaks_as_atomicpeaks:
            return fr_raw, p_raw

        # Only when needed, convert peaks to AtomicPeak (avoid repeating work in assign_wyckoff_for_msblob_peaks)
        peaks = []
        nx, ny, nz = self.F.grid.shape
        dv = float(self.F.volume/(nx*ny*nz))
        for f, val in zip(fr_raw, p_raw):
            shape, sph, elo, r_est, _ = local_shape_metrics(
                self.F.grid, f, self.F.lattice, level_rel=0.60, radius_vox=3
            )
            # Lightweight coarse integral (local-box thresholded integration; for display only; Wyckoff assignment does not depend on this)
            cx, cy, cz = [int(np.floor(fi*np.array([nx,ny,nz])[k])) for k,fi in enumerate(f)]
            rx = ry = rz = 3
            selx = np.arange(cx-rx, cx+rx+1) % nx
            sely = np.arange(cy-ry, cy+ry+1) % ny
            selz = np.arange(cz-rz, cz+rz+1) % nz
            sub = self.F.grid[np.ix_(selx, sely, selz)]
            thr = sub.max() * 0.6
            q = float(np.sum(sub[sub >= thr]) * dv)

            peaks.append(AtomicPeak(
                frac=wrap_frac(f),
                cart=frac_to_cart(wrap_frac(f), self.F.lattice),
                shape=shape,
                sphericity=float(sph),
                elongation=float(elo),
                peak_value=float(val),
                est_radius=float(r_est),
                est_charge=q
            ))
        return peaks

    # --- 2) One-shot: classification + BCP backtracking + merge + Wyckoff (delegates to the existing function) ---
    def classify_and_assign_wyckoff(self,
                                    sg: int = 221,
                                    # Pass through key thresholds; keep default consistent with assign_wyckoff_for_msblob_peaks
                                    sigmas_A=(0.40, 0.60, 0.70, 0.90, 1.20),
                                    z_mad: float = 2.5,
                                    nms_sep_A: float = 0.80,
                                    sphere_sphericity_thr: float = 0.80,
                                    atom_elong_thr: float = 1.25,
                                    bcp_keep_ratio: float = 0.90,
                                    pca_radius_A: float = 1.6,
                                    neigh_R_A: float = 1.5,
                                    s_max_A: float = 2.8,
                                    ds_A: float = 0.02,
                                    bcp_cluster_tol_A: float = 0.5,
                                    atom_merge_tol_A: float = 0.35,
                                    symprec: float = 1e-3):
        """
        Recommended to call this method to complete the entire “Step 1”:
          - msblob peak finding → atom/bond classification → BCP-based atom backtracking → PBC merging
          - spglib per-site Wyckoff assignment → final occupancy statistics
        Returns the same structure as assign_wyckoff_for_msblob_peaks.
        """
        return assign_wyckoff_for_msblob_peaks(
            self.F.grid, self.F.lattice, sg=sg,
            sigmas_A=sigmas_A, z_mad=z_mad, nms_sep_A=nms_sep_A,
            sphere_sphericity_thr=sphere_sphericity_thr, atom_elong_thr=atom_elong_thr,
            bcp_keep_ratio=bcp_keep_ratio, pca_radius_A=pca_radius_A,
            neigh_R_A=neigh_R_A, s_max_A=s_max_A, ds_A=ds_A,
            bcp_cluster_tol_A=bcp_cluster_tol_A, atom_merge_tol_A=atom_merge_tol_A,
            symprec=symprec
        )

# ---- bond-based minima atoms ----
def _pca_axis_for_peak(grid, lattice, center_frac, pca_radius_A=1.6):
    nx, ny, nz = grid.shape
    a = norm(lattice[0]); b = norm(lattice[1]); c = norm(lattice[2])
    vox = (a/nx + b/ny + c/nz)/3.0
    step = max(vox/1.2, 0.05)
    r = float(pca_radius_A)
    lin = np.arange(-r, r+1e-12, step)
    X,Y,Z = np.meshgrid(lin, lin, lin, indexing='ij')
    m = (X*X + Y*Y + Z*Z) <= (r*r + 1e-12)
    P = np.stack([X[m],Y[m],Z[m]], axis=1)
    LinvT = inv(lattice).T
    c0 = frac_to_cart(center_frac, lattice)
    fr = ((c0[None,:] + P) @ LinvT) % 1.0
    coords = (fr*np.array([nx,ny,nz], float))
    rho = ndi.map_coordinates(grid, coords.T, order=1, mode='wrap')
    w = rho - rho.min()
    if np.sum(w) <= 0: w = np.ones_like(rho)
    mu = (P * w[:,None]).sum(axis=0)/np.sum(w)
    Xc = P - mu
    C = (Xc.T * w) @ Xc / max(float(np.sum(w)), 1e-16)
    vals, vecs = eigh(C); V = vecs[:, np.argsort(vals)[::-1]]
    v1 = V[:,0]/(norm(V[:,0])+1e-16)
    return v1

# ---- step2: fill missed atoms via Wyckoff-guided local-minimum scoring ----
def _shell_mean_density(grid: np.ndarray, lattice: np.ndarray, center_frac: np.ndarray, radius_A: float, n_dirs: int = 64) -> float:
    """
    Approximate the mean density on a spherical shell of radius (Å) around center_frac.
    Uses a deterministic Fibonacci sphere sampling of directions for stability.
    """
    G = np.asarray(grid, float)
    L = np.asarray(lattice, float)
    nx, ny, nz = G.shape
    # Golden angle method for approximately uniform directions
    i = np.arange(n_dirs, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / float(n_dirs))
    theta = (np.pi * (1 + 5 ** 0.5)) * i
    # unit vectors
    u = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)
    # convert to fractional displacements
    step_cart = radius_A * u  # (n,3) in Å
    LinvT = inv(L).T
    c0 = frac_to_cart(center_frac, L)
    fr = ((c0[None, :] + step_cart) @ LinvT) % 1.0
    coords = (fr * np.array([nx, ny, nz], float))
    vals = ndi.map_coordinates(G, coords.T, order=1, mode='wrap')
    return float(np.mean(vals)) if vals.size else float(trilinear_sample(G, center_frac))


def _local_minimum_score(grid: np.ndarray, lattice: np.ndarray, center_frac: np.ndarray,
                         shells_A=(0.55, 0.90), eps: float = 1e-9) -> float:
    """
    A positive score indicates a plausible atomic-site *minimum* (valence rho dip).
    score = mean(rho on shells) - rho(center), normalized by dynamic range.
    """
    center = float(trilinear_sample(grid, center_frac))
    shell_vals = [ _shell_mean_density(grid, lattice, center_frac, r) for r in shells_A ]
    shell_mean = float(np.mean(shell_vals)) if shell_vals else center
    num = shell_mean - center  # depth of local minimum
    denom = max(shell_mean, eps)
    return float(num / denom)


def recover_atoms_via_wyckoff_minima(grid: np.ndarray,
                                     lattice: np.ndarray,
                                     sg: int,
                                     existing_atoms_frac: np.ndarray,
                                     letters_hint: Optional[List[str]] = None,
                                     # scoring / thresholds
                                     shells_A=(0.55, 0.90),
                                     min_rel_drop: float = 0.9,
                                     near_merge_tol_A: float = 0.35,
                                     # NEW: DOF sampling grid (several rational points in [0,1))
                                     param_grid: Optional[Tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75)
                                     ) -> Dict[str, Any]:
    """
    Generalized to all space groups:
      - Read Wyckoff data (spglib DB) for prefer_sg;
      - Add fixed coordinates directly as candidates; for DOF-bearing ones, sample variables (x/y/z) over param_grid to get candidates;
      - Score candidates with local-minimum metric (_local_minimum_score). Add if above threshold and not conflicting with existing atoms;
      - Merge and recompute Wyckoff (wyckoff_occupancy_from_atoms).
    """
    sg = int(sg)
    L = np.asarray(lattice, float)
    existing = wrap_frac(np.asarray(existing_atoms_frac, float).reshape(-1, 3)) \
               if existing_atoms_frac is not None else np.zeros((0,3), float)

    # ---------- helpers ----------
    def _too_close(fr):
        if existing.size == 0: return False
        dF = fr[None, :] - existing
        dF -= np.round(dF)
        dC = dF @ L
        d = np.linalg.norm(dC, axis=1).min()
        return bool(d <= float(near_merge_tol_A))

    def _extract_wy_list(info: dict):
        return (
            info.get("wyckoffs")
            or info.get("wyckoff")
            or info.get("wyckoff_positions")
            or info.get("special_positions")
            or info.get("positions")
            or []
        )

    def _extract_coords_list(w: dict):
        return (
            w.get("coordinates")
            or w.get("site_coordinates")
            or w.get("positions")
            or w.get("site_positions")
            or w.get("representative_positions")
            or []
        )

    def _parse_rational(tok: str):
        m = re.fullmatch(r"([+-]?\d+)(?:/(\d+))?", tok)
        if not m: return None
        num = int(m.group(1)); den = int(m.group(2)) if m.group(2) else 1
        return (num/den) % 1.0

    def _parse_component_expr(s: str):
        """
        "0" "1/2" "0.25" "x" "-x" "x+1/2" "1/2-x" "y-1/4" "-z+3/4" ...
        -> {'var': None/'x'/'y'/'z', 'sign': ±1 or 0, 'const': float∈[0,1)}
        """
        ss = s.replace(" ", "").lower()
        # Pure constants (try float first, then fraction)
        if re.fullmatch(r"[+-]?\d+(?:/\d+)?", ss):
            v = _parse_rational(ss)
            return {'var': None, 'sign': 0, 'const': float(v)}
        try:
            v = float(ss) % 1.0
            return {'var': None, 'sign': 0, 'const': float(v)}
        except Exception:
            pass
        var = None
        for ch in ("x", "y", "z"):
            if ch in ss: var = ch; break
        if var is None:
            return None
        parts = ss.split(var)
        before = parts[0]
        after = parts[1] if len(parts) > 1 else ""
        sign = +1
        const = 0.0
        if before.endswith("-"):
            sign = -1; before = before[:-1]
        elif before.endswith("+"):
            sign = +1; before = before[:-1]
        elif before == "-":
            sign = -1; before = ""
        elif before == "+":
            sign = +1; before = ""
        if before:
            v = _parse_rational(before)
            if v is None:
                try: v = float(before) % 1.0
                except Exception: return None
            const = (const + v) % 1.0
        if after:
            if after[0] in "+-":
                sgn = 1 if after[0] == "+" else -1
                v = _parse_rational(after[1:])
                if v is None:
                    try: v = float(after[1:]) % 1.0
                    except Exception: return None
                const = (const + sgn * v) % 1.0
            else:
                v = _parse_rational(after)
                if v is None:
                    try: v = float(after) % 1.0
                    except Exception: return None
                const = (const + v) % 1.0
        return {'var': var, 'sign': int(sign), 'const': float(const)}

    def _parse_line(coord) -> Optional[List[dict]]:
        if isinstance(coord, str):
            parts = re.split(r"[,\s]+", coord.strip())
        else:
            parts = list(map(str, coord))
        if len(parts) != 3: return None
        out=[]
        for p in parts:
            comp = _parse_component_expr(p)
            if comp is None: return None
            out.append(comp)
        return out

    def _eval_line_with_params(line: List[dict], params: Dict[str, float]) -> np.ndarray:
        q = np.zeros(3, float)
        for k in range(3):
            comp = line[k]
            if comp['var'] is None:
                q[k] = comp['const']
            else:
                v = params.get(comp['var'], 0.0)
                q[k] = (comp['sign'] * v + comp['const']) % 1.0
        return wrap_frac(q)

    # ---------- 1) Generate “all SG” candidate representatives (prefer low DOF) ----------
    trials = {}
    G = Group(int(sg))
    num_wy = len(G)
    param_grid = tuple(param_grid) if (param_grid and len(param_grid)>0) else (0.0, 0.5)
    
    # For global dedup based on identical fractional coordinates: record the *smallest DOF* seen at that position
    seen_pos_best_dof = {}  # key=(qx,qy,qz) -> int(dof)
    seen = set()            # keep the old fine-grained dedup (including wy_idx)
    
    def _eval_tok(tok, x=0.0, y=0.0, z=0.0):
        s = str(tok).strip().lower().replace(" ", "")
        s = s.replace("x", f"({x})").replace("y", f"({y})").replace("z", f"({z})")
        if not re.fullmatch(r"[0-9\.\+\-\/\(\)]+", s):  # safe whitelist
            raise ValueError(f"bad token: {tok}")
        val = eval(s, {"__builtins__": {}}, {})
        return float(val) % 1.0
    
    def _dof_of_triplet(tri) -> int:
        s = "".join(map(str, tri)).lower()
        vars_here = set([ch for ch in "xyz" if ch in s])
        return len(vars_here)
    
    # First collect (dof, wy_idx, tri) and sort by dof ascending; ensure “low DOF takes priority”
    wy_tris = []
    for i in range(num_wy):
        wy = G[num_wy - i - 1]
        wy_idx = f"{wy.multiplicity}{wy.letter}"
        if letters_hint and (wy_idx not in letters_hint and wy.letter not in letters_hint):
            continue
        triplets = _string_to_fracs(str(G.Wyckoff_positions[num_wy - i - 1]))  # → [("0","1/2","z"), ...]
        for tri in triplets:
            dof = _dof_of_triplet(tri)
            wy_tris.append((dof, wy_idx, tri))
    
    # Key: low DOF first
    wy_tris.sort(key=lambda t: t[0])
    
    for dof, wy_idx, tri in wy_tris:
        # Fixed coordinates (dof==0): add directly once
        if dof == 0:
            q = np.array([_eval_tok(tri[0]), _eval_tok(tri[1]), _eval_tok(tri[2])], float)
            pos_key = (round(q[0],6), round(q[1],6), round(q[2],6))
            # Low DOF (=0) always overrides higher DOF
            best = seen_pos_best_dof.get(pos_key, 1e9)
            if 0 <= best:
                seen_pos_best_dof[pos_key] = 0
            key = (pos_key[0], pos_key[1], pos_key[2], wy_idx)
            if key in seen: continue
            seen.add(key)
            trials.setdefault(wy_idx, []).append(wrap_frac(q))
            continue
    
        # DOF ≥ 1: sample (x,y,z); but skip if a lower DOF already occupies this position
        s = "".join(map(str, tri)).lower()
        if any(v in s for v in "xyz"):
            for x in param_grid:
                for y in param_grid:
                    for z in param_grid:
                        q = np.array([_eval_tok(tri[0], x,y,z),
                                      _eval_tok(tri[1], x,y,z),
                                      _eval_tok(tri[2], x,y,z)], float)
                        pos_key = (round(q[0],6), round(q[1],6), round(q[2],6))
                        best = seen_pos_best_dof.get(pos_key, 1e9)
                        # If a lower DOF has been recorded (best <= dof-1), skip current higher DOF candidate
                        if best <= dof:
                            continue
                        # Otherwise refresh with smaller dof
                        if dof < best:
                            seen_pos_best_dof[pos_key] = dof
                        key = (pos_key[0], pos_key[1], pos_key[2], wy_idx)
                        if key in seen: 
                            continue
                        seen.add(key)
                        trials.setdefault(wy_idx, []).append(wrap_frac(q))
    # As a safety note: no xyz means dof==0 already handled above

    # ---------- 2) Scoring + select (this part considers dof=0 already) ----------
    candidate_scores: Dict[str, List[Tuple[List[float], float]]] = {}
    added, added_scores = [], []
    for label, qs in trials.items():
        for q in qs:
            f = wrap_frac(np.asarray(q, float))
            if _too_close(f):
                continue
            s = _local_minimum_score(grid, lattice, f, shells_A=shells_A)
            candidate_scores.setdefault(label, []).append((f.tolist(), float(s)))
            if s >= float(min_rel_drop):
                added.append(f); added_scores.append(float(s))

    added = np.asarray(added, float) if len(added) else np.zeros((0,3), float)

    # ---------- 3) Merge and recompute Wyckoff ----------
    merged = merge_atom_sets(added, existing, lattice, tol_A=near_merge_tol_A)
    wy_occ, ds = wyckoff_occupancy_from_atoms(lattice, merged, symprec=1e-3, prefer_sg=sg, prefer_letter="a")
    wy_list = (list(ds.wyckoffs) if (ds is not None and hasattr(ds, "wyckoffs")) else
               (list(ds["wyckoffs"]) if (ds is not None and "wyckoffs" in ds) else []))

    return {
        "added_fracs": (added.tolist() if added.size else []),
        "added_scores": added_scores,
        "candidate_scores": candidate_scores,
        "final_atoms": (merged.tolist() if merged.size else []),
        "wyckoff_occ": wy_occ,
        "spglib_dataset": ds,
        "wyckoff_letters": wy_list,
    }


# Convenience: attach as a PeakDetector helper to chain after classify_and_assign_wyckoff
def refine_missing_via_wyckoff_minima(F: DensityField,
                                      res_step1: Dict[str, Any],
                                      sg: int,
                                      letters_hint: Optional[List[str]] = None,
                                      shells_A=(0.55, 0.90),
                                      min_rel_drop: float = 0.9,
                                      near_merge_tol_A: float = 0.35) -> Dict[str, Any]:
    """
    Usage:
        res1 = det.classify_and_assign_wyckoff(sg=221)
        res2 = refine_missing_via_wyckoff_minima(det.F, res1, sg=221, letters_hint=["1a"])
    """
    final_atoms = np.asarray(res_step1.get("final_atoms", []), float) if res_step1 else np.zeros((0,3), float)
    out = recover_atoms_via_wyckoff_minima(
        grid=F.grid, lattice=F.lattice, sg=sg, existing_atoms_frac=final_atoms,
        letters_hint=letters_hint, shells_A=shells_A, min_rel_drop=min_rel_drop, near_merge_tol_A=near_merge_tol_A
    )
    # Compose a merged result mirroring step1 structure, plus bookkeeping
    merged = dict(res_step1) if res_step1 else {}
    merged["final_atoms"] = out["final_atoms"]
    merged["final_wyckoff_occupancy"] = out["wyckoff_occ"]
    merged["spglib_sg_detected"] = dict(merged.get("spglib_sg_detected", {}))
    merged["spglib_sg_detected"]["final_after_minima"] = (int(out["spglib_dataset"].number) if (out["spglib_dataset"] is not None and hasattr(out["spglib_dataset"], "number")) else merged["spglib_sg_detected"].get("final"))
    merged["wyckoff_letters"] = out.get("wyckoff_letters")
    merged["added_by_minima"] = {
        "added_fracs": out["added_fracs"],
        "added_scores": out["added_scores"],
        "candidate_scores": out["candidate_scores"],
        "params": {
            "shells_A": list(shells_A),
            "min_rel_drop": float(min_rel_drop),
            "near_merge_tol_A": float(near_merge_tol_A),
            "letters_hint": (list(letters_hint) if letters_hint else None),
        }
    }
    return merged

def _first_min_after_90pct(grid, lattice, center_frac, axis_cart,
                           keep_ratio=0.90, s_max_A=2.8, ds_A=0.02, neigh_R_A=1.5):
    nx,ny,nz = grid.shape
    c0 = frac_to_cart(center_frac, lattice)
    a = norm(lattice[0]); b = norm(lattice[1]); c = norm(lattice[2])
    vox = (a/nx + b/ny + c/nz)/3.0
    idx_R = max(1, int(np.ceil(neigh_R_A/max(vox,1e-12))))
    rng = np.arange(-idx_R, idx_R+1)
    DX,DY,DZ = np.meshgrid(rng, rng, rng, indexing='ij')
    offs = np.stack([DX.ravel()/nx, DY.ravel()/ny, DZ.ravel()/nz], axis=1)
    frn = (center_frac[None,:] + offs) % 1.0
    cn  = frn @ lattice
    mask = np.linalg.norm(cn - c0[None,:], axis=1) <= (neigh_R_A+1e-12)
    frn = frn[mask]
    coords = (frn*np.array([nx,ny,nz], float))
    neigh_vals = ndi.map_coordinates(grid, coords.T, order=1, mode='wrap')
    rho0 = trilinear_sample(grid, center_frac)
    rho_min = float(np.min(neigh_vals)) if neigh_vals.size else rho0
    thr = rho_min + keep_ratio*(rho0 - rho_min)

    LinvT = inv(lattice).T
    s = np.arange(-float(s_max_A), float(s_max_A)+1e-12, float(ds_A))
    fr_line = ((c0[None,:] + s[:,None]*axis_cart[None,:]) @ LinvT) % 1.0
    coords = (fr_line*np.array([nx,ny,nz], float))
    rho_line = ndi.map_coordinates(grid, coords.T, order=1, mode='wrap')
    rho_s = ndi.gaussian_filter1d(rho_line, sigma=1.0)

    def side_min(positive=True):
        idx = np.where(s>=0)[0] if positive else np.where(s<=0)[0]
        ss, rr = s[idx], rho_s[idx]
        hit = np.where(rr <= thr)[0]
        if hit.size == 0:
            search_ss, search_rr = ss, rr
        else:
            k = hit[0] if positive else hit[-1]
            search_ss = ss[k:] if positive else ss[:k+1]
            search_rr = rr[k:] if positive else rr[:k+1]
        km = int(np.argmin(search_rr))
        s_min = float(search_ss[km])
        c_min = c0 + s_min*axis_cart
        return wrap_frac(cart_to_frac(c_min, lattice))

    return side_min(False), side_min(True)

def _cluster_points_frac(fracs, lattice, tol_A=0.35):
    """
    Cluster by PBC minimum-image distance: any two points with distance ≤ cutoff_A are in the same cluster (connected component).
    Returns:
      centers_frac: (K,3) fractional coordinates of each cluster center (wrapped to [0,1))
      groups      : length-K list; each item is the list of original point indices in that cluster
    Depends on helper functions implemented in this file:
      - wrap_frac
      - frac_to_cart
      - cart_to_frac
      - minimum_image_cart
    """
    fracs = wrap_frac(np.asarray(fracs, float))
    if fracs.size == 0:
        return np.zeros((0, 3)), []

    L = np.asarray(lattice, float)
    N = len(fracs)

    # --- 1) Pairwise distances under PBC minimum image ---
    # Fractional coordinate differences with minimum-image (subtract rounded values)
    dF = fracs[:, None, :] - fracs[None, :, :]
    dF -= np.round(dF)  # minimum image in fractional space
    # Convert to Cartesian differences and take norms
    dC = dF @ L
    D = np.linalg.norm(dC, axis=-1)

    # Distance ≤ cutoff → connected; set diagonal True to find components conveniently
    A = (D <= float(tol_A))
    np.fill_diagonal(A, True)

    # --- 2) Connected components -> groups ---
    visited = np.zeros(N, dtype=bool)
    groups = []
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(A[u])[0]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(comp)

    # --- Key fix: unwrapping in fractional space before averaging ---
    centers = []
    for g in groups:
        g = list(g)
        f0 = fracs[g[0]]                      # reference point (fractional)
        # For all points in the group, unwrap by minimum-image in fractional space relative to reference, then average in fractional space
        # Note: unwrap in fractional space here; do not go through Cartesian→fractional round-trips
        deltas = fracs[g] - f0                # (m,3)
        deltas -= np.round(deltas)            # minimum image (fractional)
        center_frac = f0 + deltas.mean(axis=0)
        centers.append(wrap_frac(center_frac))
    return np.asarray(centers, float), groups

def atomic_candidates_from_bcps(grid, lattice, bcp_fracs,
                                keep_ratio=0.90, s_max_A=2.8, ds_A=0.02,
                                pca_radius_A=1.6, neigh_R_A=1.5,
                                cluster_tol_A=0.5, final_merge_tol_A=0.7):
    bcp_fracs = wrap_frac(np.asarray(bcp_fracs, float))
    mins, owner = [], []
    for k, f in enumerate(bcp_fracs):
        v1 = _pca_axis_for_peak(grid, lattice, f, pca_radius_A=pca_radius_A)
        fneg, fpos = _first_min_after_90pct(grid, lattice, f, v1,
                                            keep_ratio=keep_ratio, s_max_A=s_max_A,
                                            ds_A=ds_A, neigh_R_A=neigh_R_A)
        mins += [fneg, fpos]; owner += [k, k]
    centers, groups = _cluster_points_frac(mins, lattice, tol_A=cluster_tol_A)

    multiplicities, support = [], []
    owner = np.array(owner, int)
    for g in groups:
        bcp_ids = np.unique(owner[g]).tolist()
        multiplicities.append(len(bcp_ids))
        support.append(bcp_ids)

    centers_merged, _ = _cluster_points_frac(centers, lattice, tol_A=final_merge_tol_A) if len(centers)>0 else (centers, [])
    return centers_merged, np.array(multiplicities, int), support

# ---- circumsphere fallback (optional) ----
def _circumsphere_center_4(pts):
    p1,p2,p3,p4 = pts
    A = np.vstack([2*(p2-p1), 2*(p3-p1), 2*(p4-p1)])
    b = np.array([p2@p2 - p1@p1, p3@p3 - p1@p1, p4@p1], float)
    try:
        if np.linalg.matrix_rank(A) < 3: return None
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        return c
    except Exception:
        return None

def _cluster_cart_pbc(points_cart, lattice, tol=0.30):
    pts = list(points_cart); out=[]
    while pts:
        seed = pts.pop(); cluster=[seed]; rest=[]
        for q in pts:
            if norm(minimum_image_cart(q-seed, lattice)) <= tol:
                cluster.append(q)
            else:
                rest.append(q)
        pts = rest
        c0 = cluster[0]
        unwrapped = [c0 + minimum_image_cart(p-c0, lattice) for p in cluster]
        out.append(np.mean(unwrapped, axis=0))
    return np.array(out)

def reconstruct_atoms_from_midpoints(bcp_fracs, lattice,
                                     k_nn=8, bond_len_A=2.35,
                                     radius_tol=0.25, isosphere_tol=0.12, dedup_tol=0.30):
    if len(bcp_fracs)==0: return np.zeros((0,3))
    bcp_fracs = np.asarray(bcp_fracs, float)%1.0
    carts = bcp_fracs @ lattice
    M = len(carts)
    dmat = np.zeros((M,M), float)
    for i in range(M):
        for j in range(i+1, M):
            d = norm(minimum_image_cart(carts[j]-carts[i], lattice))
            dmat[i,j]=dmat[j,i]=d
    nn_idx=[]
    for i in range(M):
        order = np.argsort(dmat[i]); kn=[j for j in order if j!=i][:k_nn]; nn_idx.append(kn)
    r0 = bond_len_A*0.5; rmin=r0*(1-radius_tol); rmax=r0*(1+radius_tol)
    cand=[]
    for i in range(M):
        mi=carts[i]; nbs=nn_idx[i]
        if len(nbs)<3: continue
        for a in range(len(nbs)-2):
            for b in range(a+1,len(nbs)-1):
                for c in range(b+1,len(nbs)):
                    ids=[i,nbs[a],nbs[b],nbs[c]]
                    base=mi
                    pts=[base + minimum_image_cart(carts[j]-base, lattice) for j in ids]
                    pts=np.array(pts)
                    cen=_circumsphere_center_4(pts)
                    if cen is None: continue
                    rs=np.array([norm(p-cen) for p in pts])
                    r=float(np.mean(rs))
                    if not(rmin<=r<=rmax): continue
                    if np.ptp(rs)>isosphere_tol: continue
                    cand.append(cen)
    if not cand: return np.zeros((0,3))
    centers=_cluster_cart_pbc(cand, lattice, tol=dedup_tol)
    return wrap_frac(cart_to_frac(centers, lattice))


from typing import Optional, Dict, List, Tuple
from collections import Counter

def wyckoff_occupancy_from_atoms(lattice, 
                                 atoms_frac,
                                 symprec: float = 1e-3,             # keep for backward-compat; not used in computation
                                 prefer_sg: Optional[int] = None,
                                 prefer_letter: Optional[str] = "a"):  # keep for backward-compat
    """
    Returns:
      occ: dict like {'a': 1, 'e': 6, ...}
      ds_min: {"number": SG, "wyckoffs": [letter per atom], "equivalent_atoms": [group_id per atom]}
    """

    SG = int(prefer_sg)
    L = np.asarray(lattice, float)

    # ---------- basic tools ----------
    def _wrap01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return x - np.floor(x)

    def _pbc_cart_dist(fr_a: np.ndarray, fr_b: np.ndarray) -> float:
        # Do nearest-image in fractional, then multiply by lattice to get Cartesian distance
        d = (fr_a - fr_b + 0.5) % 1.0 - 0.5
        # Convention: lattice is 3x3 with rows as lattice vectors -> cart = frac @ L
        cart = d @ L
        return float(np.linalg.norm(cart))

    def _parse_const(token: str):
        # Constants (support "1/2", "2/3", integers, floats)
        s = str(token).strip().lower().replace(' ', '')
        m = re.fullmatch(r'([+-]?\d+)(?:/(\d+))?', s)
        if m:
            num = int(m.group(1)); den = int(m.group(2)) if m.group(2) else 1
            return (num / den) % 1.0
        try:
            return (float(s) % 1.0)
        except Exception:
            return None

    def _eval_token(tok, params):
        """
        Supports:
          - constants: 0, 1/2, 0.25
          - variables: x|y|z, -x, x±c, c±x
        """
        s = str(tok).strip().lower().replace(' ', '')
        c = _parse_const(s)
        if c is not None:
            return c
        for var in ('x', 'y', 'z'):
            if var in s:
                parts = s.split(var)
                a = +1
                left = parts[0]
                if left.endswith('-'): a = -1; left = left[:-1]
                elif left.endswith('+'): a = +1; left = left[:-1]
                elif left == '-': a = -1; left = ''
                elif left == '+': a = +1; left = ''
                const = 0.0
                # Left/right may still be constant strings (including fractions)
                if left:
                    cc = _parse_const(left)
                    if cc is None:
                        try: cc = float(eval(left))
                        except: cc = 0.0
                    const = (const + cc) % 1.0
                right = parts[1]
                if right:
                    sign = +1
                    if right[0] == '-': sign = -1; right = right[1:]
                    elif right[0] == '+': sign = +1; right = right[1:]
                    if right:
                        cc = _parse_const(right)
                        if cc is None:
                            try: cc = float(eval(right))
                            except: cc = 0.0
                        const = (const + sign * cc) % 1.0
                v = float(params.get(var, 0.0))
                return (a * v + const) % 1.0
        return None

    def _string_to_fracs(s: str) -> List[Tuple[str, str, str]]:
        # Extract triplets from Group(...).Wyckoff_positions string
        lines = re.findall(r'^\s*([^,\n]+)\s*,\s*([^,\n]+)\s*,\s*([^,\n]+)\s*$', s, flags=re.M)
        return [(a.strip(), b.strip(), c.strip()) for (a, b, c) in lines]

    # ---------- input preprocessing ----------
    peaks = _wrap01(np.asarray(atoms_frac, float))
    if peaks.ndim != 2 or peaks.shape[1] != 3:
        return {}, {"number": SG, "wyckoffs": [], "equivalent_atoms": []}

    # ---------- read Wyckoff list for SG ----------
    G = Group(SG)
    num_wy = len(G)
    wy_dict: Dict[str, List[Tuple[str,str,str]]] = {}
    for i in range(num_wy):
        wy_idx = str(G[num_wy - i - 1].multiplicity) + G[num_wy - i - 1].letter
        wy_dict[wy_idx] = _string_to_fracs(str(G.Wyckoff_positions[num_wy - i - 1]))

    # ---------- split 0 DOF vs DOF≥1 ----------
    fixed_sites: Dict[str, np.ndarray] = {}   # '4a' -> (M,3)
    var_sites = []                             # dict(label, letter, mult, dof)

    for label, trips in wy_dict.items():
        mult = int(re.match(r'(\d+)', label).group(1))
        letter = re.search(r'([A-Za-z])$', label).group(1)

        # Collect fixed points
        numeric_points = []
        for t in trips:
            nums = [_parse_const(u) for u in t]
            if all(v is not None for v in nums):
                numeric_points.append([nums[0] % 1.0, nums[1] % 1.0, nums[2] % 1.0])
        if len(numeric_points) == mult and mult > 0:
            fixed_sites[label] = np.asarray(numeric_points, float)

        # Identify variables
        vset = set()
        for (a, b, c) in trips:
            for tok in (a, b, c):
                if _parse_const(tok) is None:
                    for ch in ('x', 'y', 'z'):
                        if ch in str(tok).lower():
                            vset.add(ch)
        if vset:
            var_sites.append(dict(label=label, letter=letter, mult=mult, dof=len(vset)))

    # ---------- hyperparameters (tune if needed) ----------
    TOL_A_FIXED = 0.25       # 0-DOF hit threshold (Å)
    FIT_TOL_A   = 0.35       # DOF≥1 acceptance threshold (Å)
    MAX_ITERS   = 16         # max outer iterations
    MAX_VALS    = 12         # max candidate values per variable

    # ---------- stage 1: exact match for 0 DOF ----------
    N = len(peaks)
    assign_letter = ['-'] * N
    group_keys = [None] * N  # (label, idx) or (label, 'fit', params_sig)

    if fixed_sites:
        for i, fr in enumerate(peaks):
            best = (None, None, float('inf'))  # (label, j, dist)
            for label, pts in fixed_sites.items():
                for j, p in enumerate(pts):
                    d = _pbc_cart_dist(fr, p)
                    if d < best[2]:
                        best = (label, j, d)
            if best[0] is not None and best[2] <= TOL_A_FIXED:
                letter = re.search(r'([A-Za-z])$', best[0]).group(1)
                assign_letter[i] = letter
                group_keys[i] = (best[0], best[1])

    # ---------- stage 2: DOF≥1 fitting (variable search + trimmed-mean scoring) ----------
    def eval_orbit(label, params):
        pts = []
        for (a, b, c) in wy_dict[label]:
            qa = _eval_token(a, params); qb = _eval_token(b, params); qc = _eval_token(c, params)
            if None in (qa, qb, qc): return None
            pts.append([qa, qb, qc])
        return np.asarray(pts, float)

    def greedy_cost_with_trim(orbit, idxs):
        M = len(orbit)
        if M == 0 or len(idxs) < M: return float('inf'), []
        used, chosen, dists = set(), [], []
        for i in range(M):
            order = sorted(idxs, key=lambda j: _pbc_cart_dist(orbit[i], peaks[j]))
            for j in order:
                if j not in used:
                    used.add(j); chosen.append(j)
                    dists.append(_pbc_cart_dist(orbit[i], peaks[j]))
                    break
            else:
                return float('inf'), []
        if len(dists) >= 2:
            mx = max(dists); trimmed = [d for d in dists if d != mx]
            if trimmed: dists = trimmed
        return float(np.mean(dists)), chosen

    it = 0
    while it < MAX_ITERS:
        it += 1
        remaining = [i for i, a in enumerate(assign_letter) if a == '-']
        if not remaining:
            break

        # Candidates: sort by DOF and multiplicity ascending; skip too-large multiplicity
        var_sites.sort(key=lambda d: (d['dof'], d['mult']))
        MAX_MULT = max(4, 2 * len(remaining))
        candidates = [s for s in var_sites if 1 <= s['dof'] and s['mult'] <= MAX_MULT and s['mult'] <= len(remaining)]
        if not candidates:
            break

        # Candidate values for variables: from remaining peaks' components, plus (v±0.5) mod 1; dedup then truncate
        raw = [float(v) % 1.0 for pid in remaining for v in peaks[pid]]
        aug = []
        for v in raw:
            aug += [v, (v+0.5) % 1.0, (v-0.5) % 1.0]
        vals = sorted({round(x, 5) for x in aug})
        if len(vals) > MAX_VALS:
            vals = vals[:MAX_VALS]

        best = dict(score=float('inf'))
        for site in candidates:
            label, letter, mult = site['label'], site['letter'], site['mult']
            # Identify variable names actually appearing in this label (only x/y/z)
            vset = sorted({ch for (a,b,c) in wy_dict[label] for tok in (a,b,c)
                           for ch in ('x','y','z') if ch in str(tok).lower() and _parse_const(tok) is None})
            if not vset:
                continue
            # Enumerate variable grid (simple Cartesian product)
            from itertools import product
            for comb in product(vals, repeat=len(vset)):
                params = dict(zip(vset, comb))
                orbit = eval_orbit(label, params)
                if orbit is None or len(orbit) != mult: 
                    continue
                sc, chosen = greedy_cost_with_trim(orbit, remaining)
                if sc < best['score']:
                    best = dict(score=sc, label=label, letter=letter, params=params, chosen=chosen)

        if best['score'] >= FIT_TOL_A or best.get('chosen') is None:
            break  # no sufficient improvement this round

        # Accept the best
        letter = best['letter']; label = best['label']; params = best['params']
        params_sig = tuple(sorted((k, round(float(v) % 1.0, 5)) for k, v in params.items()))
        for pid in best['chosen']:
            assign_letter[pid] = letter
            group_keys[pid] = (label, 'fit', params_sig)
        # Continue next iteration until no improvement or maxed out

    # ---------- summarize ----------
    counts = Counter(assign_letter)
    if '-' in counts: del counts['-']
    occ = dict(sorted(counts.items(), key=lambda kv: kv[0]))

    # Equivalent-group numbering (-1 means unmatched)
    key_to_gid: Dict[Tuple, int] = {}
    equiv = [-1] * N
    gid = 0
    for i, key in enumerate(group_keys):
        if key is None:
            continue
        if key not in key_to_gid:
            key_to_gid[key] = gid; gid += 1
        equiv[i] = key_to_gid[key]

    ds_min = {"number": SG, "wyckoffs": assign_letter, "equivalent_atoms": equiv}
    return occ, ds_min

'''
(legacy alternative version omitted for brevity; unchanged except for English comments)
'''


# ---- helpers ----
def merge_atom_sets(bond_atoms, ionic_atoms, lattice, tol_A=0.35):
    if (bond_atoms is None or len(bond_atoms)==0) and (ionic_atoms is None or len(ionic_atoms)==0):
        return np.zeros((0,3))
    if bond_atoms is None or len(bond_atoms)==0:
        base = np.asarray(ionic_atoms, float)
    elif ionic_atoms is None or len(ionic_atoms)==0:
        base = np.asarray(bond_atoms, float)
    else:
        base = np.vstack([np.asarray(bond_atoms, float), np.asarray(ionic_atoms, float)])
    centers, _ = _cluster_points_frac(base, lattice, tol_A=tol_A)
    return centers

def eq_fracs_from_equiv_groups(final_atoms_frac, ds):
    """
    final_atoms_frac: (N,3) fractional coordinates (your final_atoms)
    ds: spglib.get_symmetry_dataset(...) result
    return: List[np.ndarray], each item is the equivalent set (Mi,3) for that Wyckoff site
    """
    eq = np.asarray(ds.equivalent_atoms, int)
    groups = {}
    for i, lab in enumerate(eq):
        groups.setdefault(lab, []).append(final_atoms_frac[i])
    # Return in ascending order of group id
    return [np.asarray(v, float) for k, v in sorted(groups.items(), key=lambda x: x[0])]

# ---- main ----
def main():
    parser = argparse.ArgumentParser(description="From CHGCAR to Wyckoff occupancy (bond-based + ionic-aware)")
    parser.add_argument("--chgcar", required=True)
    parser.add_argument("--sg", type=int, required=True)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)

    # adaptive (relative threshold) params
    parser.add_argument("--peak-frac-thr", type=float, default=0.05)  # single relative threshold in [0,1]
    parser.add_argument("--min-peak-sep", type=float, default=1.0)
    parser.add_argument("--fd-scale", type=float, default=1.0)

    # new simplified top-fraction mode
    parser.add_argument("--peak-mode", choices=["adaptive","topfrac"], default="topfrac",
                        help="adaptive: relative-threshold (default); topfrac: keep top X% voxels + Å-radius clustering")
    parser.add_argument("--keep-top-frac", type=float, default=0.05,
                        help="topfrac mode: keep top X% voxels (0-1)")
    parser.add_argument("--merge-radius-A", type=float, default=0.50,
                        help="topfrac mode: cluster merge radius (Å)")
    parser.add_argument("--use-voxel-centers", action="store_true",
                        help="topfrac mode: use voxel centers (i+0.5)/N; otherwise use i/N")

    parser.add_argument("--symprec", type=float, default=1e-3)
    parser.add_argument("--out-json", default="result.json")
    parser.add_argument("--atoms-source", choices=["auto","bonds","atomic","both"], default="auto")
    # bond minima params
    parser.add_argument("--keep-ratio", type=float, default=0.90)
    parser.add_argument("--pca-radius-A", type=float, default=1.6)
    parser.add_argument("--neigh-R-A", type=float, default=1.5)
    parser.add_argument("--s-max-A", type=float, default=2.8)
    parser.add_argument("--ds-A", type=float, default=0.02)
    parser.add_argument("--cluster-tol-A", type=float, default=0.2)
    parser.add_argument("--merge-atom-tol", type=float, default=0.35)
    parser.add_argument("--fallback-bondlen-A", type=float, default=2.35)
    args = parser.parse_args()

    if spglib is None:
        print("[ERROR] spglib is required.", file=sys.stderr); sys.exit(2)
    if not HAS_PMG:
        print("[ERROR] pymatgen is required.", file=sys.stderr); sys.exit(2)

    chg = read_chgcar(args.chgcar)
    print(f"[INFO] CHGCAR grid={chg.grid.shape}, V={chg.volume:.6f} Å^3, ∫ρ dV≈{chg.total_electrons:.6f} e")
    F = DensityField(chg, smooth_sigma_vox=args.smooth_sigma)
    det = PeakDetector(F)

    # Peak detection & classification
    all_peaks = det.detect_peaks(
        peak_frac_thr=args.peak_frac_thr,
        min_peak_sep=args.min_peak_sep,
        fd_scale=args.fd_scale,
        peak_mode=args.peak_mode,
        keep_top_frac=args.keep_top_frac,
        merge_radius_A=args.merge_radius_A,
        use_voxel_centers=args.use-voxel-centers if hasattr(args, "use-voxel-centers") else args.use_voxel_centers
    )
    atom_like, bond_like = det.split_atom_bond(all_peaks)
    print(f"[INFO] peaks_total={len(all_peaks)}, atom_like={len(atom_like)}, bond_like={len(bond_like)}")

    # Decide atoms path
    atoms_from_minima = np.zeros((0,3))
    ionic_atoms = np.zeros((0,3))
    if len(bond_like) <= 2:
        # Metallic/ionic fallback: primarily use atomic peaks
        ionic_atoms = np.array([p.frac for p in atom_like]) if len(atom_like)>0 else (
                      np.array([all_peaks[0].frac]) if len(all_peaks)>0 else np.zeros((0,3)))
    else:
        # Bond-minima path
        bcp_fracs = [p.frac for p in bond_like]
        atoms_from_minima, mults, support = atomic_candidates_from_bcps(
            F.grid, F.lattice, bcp_fracs,
            keep_ratio=args.keep_ratio, s_max_A=args.s_max_A, ds_A=args.ds_A,
            pca_radius_A=args.pca_radius_A, neigh_R_A=args.neigh_R_A,
            cluster_tol_A=args.cluster_tol_A, final_merge_tol_A=0.45
        )
        ionic_atoms = np.array([p.frac for p in atom_like]) if len(atom_like)>0 else np.zeros((0,3))

    print(f"[INFO] atoms_from_minima={len(atoms_from_minima)}, ionic_atoms={0 if ionic_atoms.size==0 else len(ionic_atoms)}", flush=True)

    # Decide final_atoms
    atoms_source_used = args.atoms_source
    if args.atoms_source == "bonds":
        final_atoms = atoms_from_minima
    elif args.atoms_source == "atomic":
        final_atoms = ionic_atoms
    elif args.atoms_source == "both":
        final_atoms = merge_atom_sets(atoms_from_minima, ionic_atoms, F.lattice, tol_A=args.merge_atom_tol)
    else:  # auto
        if len(bond_like) > 2 and atoms_from_minima.size:
            final_atoms = merge_atom_sets(atoms_from_minima, ionic_atoms, F.lattice, tol_A=args.merge_atom_tol)
            atoms_source_used = "bonds(auto)+atomic"
        else:
            final_atoms = ionic_atoms
            atoms_source_used = "atomic(auto)"

    merged_for_wyckoff = (0 if final_atoms.size==0 else len(final_atoms))
    print(f"[INFO] merged_for_wyckoff={merged_for_wyckoff}")

    # Wyckoff occupancy (prefer sg & letter)
    wy_occ, ds = wyckoff_occupancy_from_atoms(F.lattice, final_atoms, symprec=args.symprec, prefer_sg=args.sg, prefer_letter="a")
    sg_found = (int(ds.number) if (ds is not None and hasattr(ds,"number")) else (int(ds["number"]) if (ds is not None and "number" in ds) else None))
    print(f"[INFO] wyckoff_occ={wy_occ}, spglib_sg={sg_found}")

    # JSON
    def peak_to_json(p: AtomicPeak) -> Dict[str, Any]:
        return {
            "frac": p.frac.tolist(),
            "cart": p.cart.tolist(),
            "shape": p.shape,
            "sphericity": float(p.sphericity),
            "elongation": float(p.elongation),
            "peak_value": float(p.peak_value),
            "est_radius": float(p.est_radius),
            "est_charge": float(p.est_charge),
        }

    wy_list = (list(ds.wyckoffs) if (ds is not None and hasattr(ds,"wyckoffs")) else (list(ds["wyckoffs"]) if (ds is not None and "wyckoffs" in ds) else None))
    equiv = (list(map(int, ds.equivalent_atoms)) if (ds is not None and hasattr(ds,"equivalent_atoms")) else (list(map(int, ds["equivalent_atoms"])) if (ds is not None and "equivalent_atoms" in ds) else None))

    out = {
        "input": {
            "chgcar": args.chgcar,
            "grid": list(F.shape),
            "volume": F.volume,
            "total_electrons": F.total_electrons,
            "spacegroup_given": args.sg,
            "symprec": args.symprec,
        },
        "step1": {
            "peaks_total": len(all_peaks),
            "atom_like": len(atom_like),
            "bond_like": len(bond_like),
            "all_peaks": [peak_to_json(p) for p in all_peaks],
        },
        "atoms": {
            "atoms_source_used": atoms_source_used,
            "atoms_from_bonds": (atoms_from_minima.tolist() if atoms_from_minima.size else []),
            "atoms_from_atomic": (ionic_atoms.tolist() if ionic_atoms.size else []),
            "final_atoms": (final_atoms.tolist() if final_atoms.size else []),
        },
        "wyckoff": {
            "wyckoff_occupancy": wy_occ,
            "spglib_sg": sg_found,
            "spglib_wyckoffs": wy_list,
            "equiv_atoms": equiv
        }
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[INFO] written {args.out_json}")
    
    eq_fracs_list = eq_fracs_from_equiv_groups(final_atoms, ds)

    model = SitePatchCNN(embed_dim=64).eval()
    site_embeddings = []
    with torch.no_grad():
        for eq_fracs in eq_fracs_list:
            patch = site_patch_mean_over_equivalents(
                rho=F.grid, lattice=F.lattice, eq_fracs=eq_fracs, radius=1.0, out_size=32
            )
            emb = model(torch.from_numpy(patch[None, None])).cpu().numpy().reshape(-1)
            site_embeddings.append(emb)
    site_embeddings = np.stack(site_embeddings, 0)
    #print(site_embeddings)

if __name__ == "__main__":
    main()

