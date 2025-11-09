#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wyckoff-site 3D CNN embedding input (R=1.0 Å window, 32^3 voxels) with
multiplicity handling by averaging over *equivalent points*.

Assumptions
-----------
- You provide the charge density `rho` as a (Nx,Ny,Nz) float32 numpy array.
- `lattice` is a (3,3) numpy array (Å), rows are a,b,c in Cartesian.
- Each site is represented by a list of *equivalent fractional coords*
  (one representative for each equivalent position of that Wyckoff site).
- Voxel pitch target: 0.062 Å → edge≈ 2*R = 2.0 Å ⇒ 2.0 / 0.062 ≈ 32.26
  We fix `out_size = 32` as requested (≈ 1.984 Å window). This is close and
  consistent with your requirement to use 32^3 voxels.

What you get
------------
- `sample_site_patch(...)` : periodic trilinear interpolation around one center
- `site_patch_mean_over_equivalents(...)` : average the patches of all
   equivalent points for a Wyckoff site to form a *single* CNN input
- `SitePatchCNN` : a compact 3D CNN that maps (1,32,32,32) → embedding

Usage sketch
------------
    patch = site_patch_mean_over_equivalents(
        rho, lattice, eq_fracs=site_equiv_fracs, radius=1.0, out_size=32
    )  # (32,32,32) float32

    model = SitePatchCNN(embed_dim=64)
    emb = model(torch.from_numpy(patch[None,None]))  # (1,64)
"""

from typing import Iterable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Sampling utilities ----------------------------- #

def _frac_wrap(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coords to [0,1)^3."""
    return frac - np.floor(frac)


def _trilinear_periodic_sample(rho: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """
    Periodic trilinear interpolation on a 3D grid.

    Args:
        rho: (Nx,Ny,Nz) float32 array.
        ijk: (..., 3) array of *continuous* index coordinates in grid units (i ∈ [0,Nx), etc.).

    Returns:
        values: (...) float32
    """
    Nx, Ny, Nz = rho.shape
    i = ijk[..., 0]; j = ijk[..., 1]; k = ijk[..., 2]

    i0 = np.floor(i).astype(np.int64)
    j0 = np.floor(j).astype(np.int64)
    k0 = np.floor(k).astype(np.int64)

    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    k1 = (k0 + 1) % Nz

    di = (i - i0).astype(np.float32)
    dj = (j - j0).astype(np.float32)
    dk = (k - k0).astype(np.float32)

    i0 %= Nx; j0 %= Ny; k0 %= Nz

    c000 = rho[i0, j0, k0]
    c100 = rho[i1, j0, k0]
    c010 = rho[i0, j1, k0]
    c110 = rho[i1, j1, k0]
    c001 = rho[i0, j0, k1]
    c101 = rho[i1, j0, k1]
    c011 = rho[i0, j1, k1]
    c111 = rho[i1, j1, k1]

    c00 = c000*(1-di) + c100*di
    c10 = c010*(1-di) + c110*di
    c01 = c001*(1-di) + c101*di
    c11 = c011*(1-di) + c111*di

    c0 = c00*(1-dj) + c10*dj
    c1 = c01*(1-dj) + c11*dj

    c = c0*(1-dk) + c1*dk
    return c.astype(np.float32)


def sample_site_patch(
    rho: np.ndarray,
    lattice: np.ndarray,
    frac_center: np.ndarray,
    radius: float = 1.0,
    out_size: int = 32,
) -> np.ndarray:
    """
    Extract a cubic patch centered at `frac_center`, covering a Cartesian cube of edge ~2*radius (Å),
    resampled to (out_size, out_size, out_size) with periodic trilinear interpolation.

    Notes:
    - With voxel pitch target 0.062 Å and R=1.0 Å, out_size=32 matches your spec.

    Args:
        rho: (Nx,Ny,Nz) float32 charge density grid.
        lattice: (3,3) float32 lattice matrix in Å (rows: a,b,c).
        frac_center: (3,) float fractional coordinate.
        radius: half-edge length in Å (default 1.0).
        out_size: output resolution per axis (default 32).

    Returns:
        patch: (out_size, out_size, out_size) float32
    """
    assert rho.ndim == 3 and lattice.shape == (3,3)
    Nx, Ny, Nz = rho.shape
    fc = _frac_wrap(np.asarray(frac_center, dtype=np.float32))
    lat = np.asarray(lattice, dtype=np.float32)

    S = int(out_size)
    # Cartesian offsets spanning [-R, R]
    lin = np.linspace(-radius, radius, S, dtype=np.float32)
    dx, dy, dz = np.meshgrid(lin, lin, lin, indexing="ij")  # each (S,S,S)

    # Convert Cartesian offsets → fractional offsets using inv(lattice)
    inv_lat = np.linalg.inv(lat).astype(np.float32)
    offsets_cart = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)        # (S^3,3)
    offsets_frac = offsets_cart @ inv_lat.T                               # (S^3,3)

    # Absolute fractional positions (periodic)
    frac_pts = _frac_wrap(fc[None, :] + offsets_frac)                     # (S^3,3)

    # Map to grid index space
    ijk = np.empty_like(frac_pts, dtype=np.float64)
    ijk[:, 0] = frac_pts[:, 0] * Nx
    ijk[:, 1] = frac_pts[:, 1] * Ny
    ijk[:, 2] = frac_pts[:, 2] * Nz

    vals = _trilinear_periodic_sample(rho, ijk).reshape(S, S, S)
    return vals  # float32


def site_patch_mean_over_equivalents(
    rho: np.ndarray,
    lattice: np.ndarray,
    eq_fracs: Iterable[np.ndarray],
    radius: float = 1.0,
    out_size: int = 32,
) -> np.ndarray:
    """
    Average patches over all *equivalent fractional positions* of a Wyckoff site.

    Args:
        rho: (Nx,Ny,Nz) float32
        lattice: (3,3) float32
        eq_fracs: iterable of fractional coords for this Wyckoff site (length = multiplicity)
        radius: 1.0 Å (default)
        out_size: 32 (default)

    Returns:
        mean_patch: (out_size, out_size, out_size) float32
    """
    eq_fracs = list(eq_fracs)
    assert len(eq_fracs) >= 1, "eq_fracs must contain at least one equivalent position"

    acc = None
    for f in eq_fracs:
        p = sample_site_patch(rho, lattice, np.asarray(f, dtype=np.float32), radius=radius, out_size=out_size)
        if acc is None:
            acc = p.astype(np.float32)
        else:
            acc += p.astype(np.float32)
    mean_patch = acc / float(len(eq_fracs))
    return mean_patch.astype(np.float32)


# ----------------------------- Site embedding CNN ----------------------------- #

def _gn(c: int) -> nn.GroupNorm:
    # Choose the largest group number that divides the channel count evenly to ensure stability
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)

class SE3D(nn.Module):
    """Squeeze-and-Excitation for 3D feature maps."""
    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.fc1 = nn.Conv3d(c, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, c, kernel_size=1)

    def forward(self, x):
        # Global average pooling to 1x1x1
        w = x.mean(dim=[2, 3, 4], keepdim=True)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ResBlock3D(nn.Module):
    """
    Residual block: Conv3d -> GN -> SiLU -> Conv3d -> GN -> (SE) -> residual
    Supports downsampling by stride in the first conv layer;
    uses a 1x1x1 shortcut when input and output channels differ.
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1, use_se: bool = True, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _gn(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = _gn(out_c)
        self.se = SE3D(out_c) if use_se else nn.Identity()
        self.drop = nn.Dropout3d(p=drop) if drop > 0 else nn.Identity()

        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                _gn(out_c),
            ) if (stride != 1 or in_c != out_c) else nn.Identity()
        )

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = F.silu(self.norm1(out))
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.drop(out)
        out = F.silu(out + identity)
        return out

class SitePatchCNN(nn.Module):
    """
    Enhanced 3D CNN:
      - stem + 3 stages (each stage has 2 residual blocks; the first block uses stride=2 for downsampling)
      - Channel configuration: 32 -> 64 -> 128 -> 256
      - GroupNorm + SiLU, SE attention, projection to embed_dim, and L2 normalization
    Input:  (N, in_channels, 32, 32, 32)
    Output: (N, embed_dim)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        in_channels: int = 1,
        width: int = 32,          # Stem width; global capacity knob
        se_reduction: int = 8,
        drop: float = 0.0         # Dropout3d in residual blocks; try 0.1
    ):
        super().__init__()
        # stem: lightweight entry
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False),
            _gn(width),
            nn.SiLU(inplace=True),
        )

        # stages: double the channels at each level, downsample at the first block
        c1, c2, c3 = width, width * 2, width * 4  # 32, 64, 128 (if width=32)
        c4 = width * 8                             # 256

        self.stage1 = nn.Sequential(
            ResBlock3D(c1, c2, stride=2, use_se=True, drop=drop),  # 32 -> 16
            ResBlock3D(c2, c2, stride=1, use_se=True, drop=drop),
        )
        self.stage2 = nn.Sequential(
            ResBlock3D(c2, c3, stride=2, use_se=True, drop=drop),  # 16 -> 8
            ResBlock3D(c3, c3, stride=1, use_se=True, drop=drop),
        )
        self.stage3 = nn.Sequential(
            ResBlock3D(c3, c4, stride=2, use_se=True, drop=drop),  # 8 -> 4
            ResBlock3D(c4, c4, stride=1, use_se=True, drop=drop),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),       # -> (N, C, 1,1,1)
            nn.Flatten(1),
            nn.Linear(c4, embed_dim),
        )

        # Linear layer initialization
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

        self.se_reduction = se_reduction  # keep parameter for external inspection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, 32, 32, 32)
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        z = self.head(h)                  # (N, embed_dim)
        return F.normalize(z, dim=-1)

# ----------------------------- Smoke test (optional) ----------------------------- #
if __name__ == "__main__":
    # Synthetic example
    Nx = Ny = Nz = 86
    rho = np.random.rand(Nx,Ny,Nz).astype(np.float32)
    lattice = np.eye(3, dtype=np.float32) * 5.43

    # Suppose a Wyckoff site of multiplicity 4 (e.g., tetrahedral equivalents)
    eq_fracs = [
        np.array([0.125, 0.125, 0.125], dtype=np.float32),
        np.array([0.875, 0.875, 0.125], dtype=np.float32),
        np.array([0.875, 0.125, 0.875], dtype=np.float32),
        np.array([0.125, 0.875, 0.875], dtype=np.float32),
    ]

    patch = site_patch_mean_over_equivalents(rho, lattice, eq_fracs, radius=1.0, out_size=32)
    print("[TEST] mean patch:", patch.shape, patch.dtype, f"min={patch.min():.3g} max={patch.max():.3g}")

    model = SitePatchCNN(embed_dim=64)
    with torch.no_grad():
        emb = model(torch.from_numpy(patch[None, None]))
    print("[TEST] emb:", tuple(emb.shape))

