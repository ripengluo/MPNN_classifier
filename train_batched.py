#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP batched training with 2-stage schedule + gradient accumulation.

Key additions over your version:
- Gradient accumulation via --accum-steps (DDP-aware with no_sync()).
- Optional gradient clipping via --clip-grad-norm.
- Effective batch size per optimizer step:
      effective = batch_size * accum_steps * world_size
- Refactor: _compute_loss_on_batchlist() returns loss; stepping is handled in the loops.

Original features kept:
- Two-phase training: (1) pretrain on mixed; (2) finetune on cubic.
- Separate LRs per phase; rank-0-only logging/validation/checkpointing.
- Robustness fixes & CPU-only DDP fallback.

Usage example:
  torchrun --nproc_per_node=4 train_batched.py \
      --data-dir ./dataset --nfiles 32 \
      --batch-size 8 --accum-steps 8 \
      --pretrain-steps 5000 --ft-epochs 10 \
      --pre-lr 1e-3 --ft-lr 5e-4
"""

import os, sys, json, argparse, itertools, contextlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tqdm import tqdm

# Make sure we can import your original module
sys.path.append(os.path.dirname(__file__) or ".")
import train as base  # your original dataset/model/helpers

# ---------------- batched site-embedding ----------------
def build_node_features_batched(rho, fr_all, lattice, wyckoff_info, cnn_model, device):
    """
    Vectorized version of base.build_node_features:
      - builds all site patches, stacks to (N,1,32,32,32)
      - single CNN forward to get (N, embed_dim)
      - concatenates Wyckoff one-hot (26) + SG one-hot (230)
    """
    wy_letters = list(wyckoff_info.get("wyckoffs") or [])
    N = len(fr_all)
    sg_num = wyckoff_info.get("number")
    sg_vec = torch.from_numpy(base.sg_one_hot(int(sg_num))).to(device)  # (230,)

    # Build all 3D patches on CPU (numpy), then one batched forward on GPU
    patches = []
    L = lattice.astype(np.float32)
    rho32 = rho.astype(np.float32, copy=False)
    for f in fr_all:
        patch = base.site_patch_mean_over_equivalents(
            rho=rho32,
            lattice=L,
            eq_fracs=np.asarray([f], dtype=np.float32),
            radius=1.0, out_size=32,
        )  # (32,32,32) numpy
        patches.append(patch)

    if len(patches) == 0:
        return None

    P = torch.from_numpy(np.stack(patches)).unsqueeze(1)  # (N,1,32,32,32)
    P = P.to(device, non_blocking=True)

    Z = cnn_model(P)  # (N, embed_dim)

    wy_oh = torch.from_numpy(np.stack([base.one_hot_wy(w) for w in wy_letters])).to(device)  # (N,26)
    sg_oh = sg_vec.unsqueeze(0).repeat(N, 1)  # (N,230)
    X = torch.cat([Z, wy_oh, sg_oh], dim=-1)  # (N, embed_dim+26+230)
    return X

# Keep collate simple: return list of items as-is (graphs have variable sizes)
def _collate_keep_list(batch):
    return batch

# ---------------- DDP utils ----------------
def setup_ddp(rank: int, world_size: int, backend: str = None):
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    # set CUDA device before init (if NCCL)
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------- helper: load or build cubic index cache ----------
def _load_or_build_cubic_index(args, dataset_raw, rng, rank: int = 0):
    idx_cache = os.path.join(args.data_dir, "cubic_229_idx.txt")
    cubic_idx = []
    if os.path.exists(idx_cache):
        with open(idx_cache, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    j = int(s)
                except ValueError:
                    continue
                if 0 <= j < len(dataset_raw):
                    cubic_idx.append(j)
    else:
        # enumerate once to determine cubic
        all_idx = list(range(len(dataset_raw)))
        if args.max_samples >= 0:
            all_idx = all_idx[:min(len(all_idx), args.max_samples)]
        iterator = tqdm(all_idx, desc="Scanning for cubic", leave=False, dynamic_ncols=True) if rank == 0 else all_idx
        for i in iterator:
            try:
                item = dataset_raw[i]
                sg_num = int(item.get("spacegroup_number", ""))
                if item.get("c_system", "") == "cubic" and (sg_num == 229 or sg_num ==225 or sg_num == 221) :
                    cubic_idx.append(i)
            except Exception:
                continue
        if len(cubic_idx) == 0:
            raise SystemExit("No cubic samples found.")
        if rank == 0:
            os.makedirs(args.data_dir, exist_ok=True)
            with open(idx_cache, "w", encoding="utf-8") as f:
                f.write("\n".join(str(i) for i in cubic_idx))
            print(f"[INFO] saved {len(cubic_idx)} cubic indices to {idx_cache}")
    all_idx = list(range(len(dataset_raw)))
    cubic_set = set(cubic_idx)
    non_cubic_idx = [i for i in all_idx if i not in cubic_set]
    rng.shuffle(non_cubic_idx)
    rng.shuffle(cubic_idx)
    split = int(round((1.0 - args.test_ratio) * len(cubic_idx)))
    train_idx = cubic_idx[:split]
    test_idx  = cubic_idx[split:]
    if rank == 0:
        print(f"[INFO] cubic samples: {len(cubic_idx)}  -> train: {len(train_idx)}, test: {len(test_idx)}, non-cubic: {len(non_cubic_idx)}")
    return train_idx, test_idx, non_cubic_idx

# ---------------- forward-only loss on a batch list ----------------
def _compute_loss_on_batchlist(batch, device, cnn_model, mpnn, edge_mode):
    """
    Builds a mega-graph for the list of items and returns a scalar CE loss tensor.
    Returns None if the batch yields no valid samples.
    """
    X_list, ei_list, ea_list, y_list = [], [], [], []
    node_offset = 0

    for item in batch:
        # coeff -> rho
        rho = base.invert_coeff_to_rho(item["coeff"], out_shape=None, smooth_sigma_vox=0.0)

        # graph & features
        fr_all  = item.get("fr_all")
        wy_info = item.get("wy_info")
        if fr_all is None or wy_info is None or len(fr_all) == 0:
            continue

        G = base.build_graph(item["lattice"], fr_all, nn_buffer_A=0.8, edge_mode=edge_mode)
        Xi = build_node_features_batched(rho, fr_all, item["lattice"], wy_info, cnn_model, device)
        if Xi is None or Xi.size(0) == 0:
            continue

        # labels (Z<=83 only; same as original logic)
        elem_map   = item.get("elem_map", {})
        wy_letters = wy_info.get("wyckoffs") or []
        if any((w not in elem_map) for w in wy_letters):
            continue
        y_symbols = [elem_map[w] for w in wy_letters]
        if any((s not in base.sym2Z) for s in y_symbols):
            continue
        yi = torch.tensor([base.Element(s).Z - 1 for s in y_symbols],
                          dtype=torch.long, device=device)  # (N,)

        # edges (offset for batching)
        if "edge_index" in G and getattr(G["edge_index"], "size", 0) > 0:
            ei = torch.from_numpy(G["edge_index"].copy()).long()
            ei += node_offset
            ea = torch.from_numpy(G["edge_attr"].copy()).float()
        else:
            ei = torch.zeros((2,0), dtype=torch.long)
            ea = torch.zeros((0,3), dtype=torch.float32)
        node_offset += Xi.size(0)

        X_list.append(Xi)
        ei_list.append(ei)
        ea_list.append(ea)
        y_list.append(yi)

    if node_offset == 0:
        return None  # skip (no valid samples)

    # concat mega-graph
    X = torch.cat(X_list, dim=0).to(device, non_blocking=True)
    edge_index = (torch.cat(ei_list, dim=1).to(device, non_blocking=True)
                  if ei_list else torch.zeros((2,0), dtype=torch.long, device=device))
    edge_attr = (torch.cat(ea_list, dim=0).to(device, non_blocking=True)
                 if ea_list else torch.zeros((0,3), dtype=torch.float32, device=device))
    y = torch.cat(y_list, dim=0).to(device, non_blocking=True)

    logits = mpnn(X, edge_index, edge_attr)
    loss = F.cross_entropy(logits, y)
    return loss  # tensor on device

# ---------------- worker (per-rank) ----------------
def train_worker(rank: int, world_size: int, args):
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    setup_ddp(rank, world_size, backend=backend)

    # System & RNG
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed + rank)

    # ---------- datasets ----------
    h5_paths = [os.path.join(args.data_dir, f"mp_chg_{i:03d}.h5") for i in range(args.nfiles)]
    datasets = [base.H5CoeffDataset(p) for p in h5_paths if os.path.exists(p)]
    if not datasets:
        if rank == 0:
            print(f"[FATAL] No H5 files found under {args.data_dir}")
        cleanup_ddp()
        return
    dataset_raw = base.ConcatDataset(datasets)

    # split train/test (deterministic)
    train_idx, test_idx, non_cubic_idx = _load_or_build_cubic_index(args, dataset_raw, rng, rank=rank)
    cubic_train_set = Subset(dataset_raw, train_idx)
    test_set  = Subset(dataset_raw, test_idx)  # only rank 0 validates
    mixed_train_set = Subset(dataset_raw, train_idx + non_cubic_idx)

    # ---------- distributed sampler + dataloader ----------
    sampler_cubic = DistributedSampler(
        cubic_train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    sampler_mixed = DistributedSampler(
        mixed_train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    dl_common = dict(
        batch_size=int(args.batch_size),
        shuffle=False,  # handled by DistributedSampler
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers) and int(args.num_workers) > 0,
        collate_fn=_collate_keep_list,
        drop_last=False,
    )
    if int(args.num_workers) > 0:
        dl_common["prefetch_factor"] = int(args.prefetch_factor)

    mixed_loader = DataLoader(mixed_train_set, sampler=sampler_mixed, **dl_common)
    cubic_loader = DataLoader(cubic_train_set, sampler=sampler_cubic, **dl_common)

    # ---------- models ----------
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    embed_dim = 128
    cnn_model = base.SitePatchCNN(embed_dim=embed_dim).to(device)
    in_dim = embed_dim + 26 + 230   # CNN + Wy one-hot + SG one-hot
    edge_dim = 3
    hidden = 256
    #mpnn = base.MPNN(in_dim=in_dim, edge_dim=edge_dim, hidden=hidden, num_layers=5, wy_dim=26, wy_start=embed_dim, num_classes=83).to(device)
    mpnn = base.MPNN(in_dim=in_dim, edge_dim=edge_dim, hidden=hidden, num_layers=4, num_classes=83).to(device)

    # Wrap with DDP
    find_unused = False
    cnn_model = DDP(cnn_model, device_ids=[rank] if device.type == "cuda" else None,
                    output_device=rank if device.type == "cuda" else None,
                    find_unused_parameters=find_unused)
    mpnn      = DDP(mpnn,      device_ids=[rank] if device.type == "cuda" else None,
                    output_device=rank if device.type == "cuda" else None,
                    find_unused_parameters=find_unused)

    # Params & optimizer
    if rank == 0:
        n_params_cnn  = sum(p.numel() for p in cnn_model.module.parameters())
        n_params_mpnn = sum(p.numel() for p in mpnn.module.parameters())
        print(f"CNN Params:  {n_params_cnn/1e6:.2f} M")
        print(f"MPNN Params: {n_params_mpnn/1e6:.2f} M")

    optimizer = torch.optim.Adam(
        itertools.chain(mpnn.parameters(), cnn_model.parameters()), lr=args.pre_lr
    )

    # --- resume (optional) ---
    resume_path = args.resume_from
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"[RESUME] loading checkpoint from: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        args.pretrain_steps = 0
        mpnn.module.load_state_dict(ckpt["mpnn_state_dict"], strict=False)
        cnn_model.module.load_state_dict(ckpt["cnn_state_dict"], strict=False)
        if args.strict_resume and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                if rank == 0:
                    print(f"[RESUME] optimizer state incompatible, skipping. error={e}")
        for g in optimizer.param_groups:
            g["lr"] = float(args.ft_lr)
        if rank == 0:
            print(f"[RESUME] weights loaded. Switched LR to finetune lr={args.ft_lr}")

    # ---- show effective batch size per optimizer step (rank-0) ----
    if rank == 0:
        eff = int(args.batch_size) * max(1, int(args.accum_steps)) * int(world_size)
        print(f"[EFFECTIVE] graphs/optimizer_step = batch_size({args.batch_size}) "
              f"* accum_steps({args.accum_steps}) * world_size({world_size}) = {eff}")

    # logs & ckpt (rank 0 only)
    best_val_loss = float("inf")
    ckpt_dir = "checkpoints"
    pretrain_ckpt = os.path.join(ckpt_dir, "pretrain_last.pt")
    finetune_best = os.path.join(ckpt_dir, "finetune_best.pt")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open("log_train", "w") as f:
            print(f"[INFO] Datasets -> mixed: {len(mixed_train_set)} | cubic_train: {len(cubic_train_set)} | cubic_test: {len(test_set)}", file=f)

    # --------------- Phase 1: PRETRAIN on mixed ----------------
    steps_left = int(args.pretrain_steps)
    pt_epoch = 0
    accum_steps = max(1, int(args.accum_steps))
    do_clip = float(args.clip_grad_norm) > 0.0
    params_for_clip = list(itertools.chain(cnn_model.parameters(), mpnn.parameters()))

    if steps_left > 0:
        cnn_model.train(); mpnn.train()
        loss_sum_local, steps_count_local = 0.0, 0

        optimizer.zero_grad(set_to_none=True)
        valid_in_group = 0  # counts valid micro-batches in current accumulation group

        while steps_left > 0:
            pt_epoch += 1
            sampler_mixed.set_epoch(pt_epoch)
            pbar = tqdm(mixed_loader, desc=f"Pretrain epoch {pt_epoch}", dynamic_ncols=True) if rank == 0 else mixed_loader

            for batch in pbar:
                loss = _compute_loss_on_batchlist(
                    batch=batch, device=device, cnn_model=cnn_model, mpnn=mpnn, edge_mode=args.edge_mode
                )
                if loss is None:
                    # skip invalid micro-batch without touching accumulation
                    continue

                # Decide whether to all-reduce on this micro-batch (only on the last of the group)
                valid_in_group += 1
                steps_left -= 1
                should_sync = (valid_in_group % accum_steps == 0)
                #if rank == 0:
                #    with open("log_train", "a") as _f:
                #        print("steps_left", steps_left, file=_f)

                with contextlib.ExitStack() as stack:
                    if not should_sync:
                        stack.enter_context(cnn_model.no_sync())
                        stack.enter_context(mpnn.no_sync())
                    (loss / accum_steps).backward()

                if should_sync:
                    if do_clip:
                        torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.clip_grad_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # bookkeeping per *optimizer step*
                    steps_count_local += 1
                    loss_sum_local += float(loss.item())
                    valid_in_group = 0

                    if rank == 0:
                        pbar.set_postfix({"loss": f"{loss.item():.4f}", "steps_left": steps_left})

                if steps_left <= 0:
                    break

    
            if rank == 0:
                with torch.no_grad():
                    val_loss = base._compute_validation_loss(
                        test_set=test_set, device=device,
                        cnn_model=cnn_model.module, mpnn=mpnn.module,
                        ce_loss=F.cross_entropy,
                        max_samples=int(args.val_max_samples)
                    )
                with open("log_train", "a") as _f:
                    print(f"[E{pt_epoch}] pretrain total mean loss = {loss_sum_local / max(1, steps_count_local):.6f}", file=_f)
                    print(f"[VAL@finetune epoch {epoch}] mean validation loss = {val_loss:.6f}", file=_f)
    
                # save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.dirname(finetune_best) or ".", exist_ok=True)
                    torch.save({
                        "phase": "finetune",
                        "epoch": epoch,
                        "val_loss": float(val_loss),
                        "mpnn_state_dict": mpnn.module.state_dict(),
                        "cnn_state_dict": cnn_model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    }, finetune_best)
                    msg = f"[BEST] epoch={epoch}  val_loss={val_loss:.6f}  -> saved to {finetune_best}"
                    with open("log_train", "a") as _f:
                        print(msg, file=_f)
    



    # --------------- switch LR for finetune ----------------
    for g in optimizer.param_groups:
        g["lr"] = float(args.ft_lr)

    # --------------- Phase 2: FINETUNE on cubic ----------------
    best_val_loss = float("inf")
    if rank == 0:
        print(f"[FINETUNE] epochs={args.ft_epochs}  lr={args.ft_lr}  accum_steps={accum_steps}")

    for epoch in range(1, int(args.ft_epochs) + 1):
        cnn_model.train(); mpnn.train()
        sampler_cubic.set_epoch(epoch)
        loss_sum_local, steps_count_local = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        valid_in_group = 0

        pbar = tqdm(cubic_loader, desc=f"Finetune epoch {epoch}", dynamic_ncols=True) if rank == 0 else cubic_loader

        for batch in pbar:
            loss = _compute_loss_on_batchlist(
                batch=batch, device=device, cnn_model=cnn_model, mpnn=mpnn, edge_mode=args.edge_mode
            )
            if loss is None:
                continue

            valid_in_group += 1
            should_sync = (valid_in_group % accum_steps == 0)

            with contextlib.ExitStack() as stack:
                if not should_sync:
                    stack.enter_context(cnn_model.no_sync())
                    stack.enter_context(mpnn.no_sync())
                (loss / accum_steps).backward()

            if should_sync:
                if do_clip:
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.clip_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                steps_count_local += 1
                loss_sum_local += float(loss.item())
                valid_in_group = 0

                if rank == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # NOTE: we intentionally do not force a partial step for leftover micro-batches
        # to keep gradient scale consistent; leftover grads carry into the next epoch.

        # ---- reduce running stats across ranks ----
        stats = torch.tensor([loss_sum_local, steps_count_local], dtype=torch.float32, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_steps = max(1.0, stats[1].item())
        if rank == 0:
            mean_loss = total_loss / total_steps
            print(f"[E{epoch}] finetune mean loss (avg across ranks)={mean_loss:.6f}")

        # ---- validation (rank 0 only) ----
        if rank == 0:
            with torch.no_grad():
                val_loss = base._compute_validation_loss(
                    test_set=test_set, device=device,
                    cnn_model=cnn_model.module, mpnn=mpnn.module,
                    ce_loss=F.cross_entropy,
                    max_samples=int(args.val_max_samples)
                )
            with open("log_train", "a") as _f:
                print(f"[VAL@finetune epoch {epoch}] mean validation loss = {val_loss:.6f}", file=_f)

            # save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(finetune_best) or ".", exist_ok=True)
                torch.save({
                    "phase": "finetune",
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "mpnn_state_dict": mpnn.module.state_dict(),
                    "cnn_state_dict": cnn_model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                }, finetune_best)
                msg = f"[BEST] epoch={epoch}  val_loss={val_loss:.6f}  -> saved to {finetune_best}"
                with open("log_train", "a") as _f:
                    print(msg, file=_f)

        dist.barrier()

    # ---------- save final (rank 0 only) ----------
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(mpnn.module.state_dict(), "mpnn_site_cls.pt")
        with open("label_space.json", "w", encoding="utf-8") as f:
            json.dump({"id2elem": []}, f, indent=2, ensure_ascii=False)
        print("✅ Training finished. Saved model to mpnn_site_cls.pt and label_space.json")

    cleanup_ddp()

def parse_args():
    ap = argparse.ArgumentParser()
    # ---------- original knobs ----------
    ap.add_argument("--data-dir", type=str, default="./dataset")
    ap.add_argument("--nfiles", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-3)  # kept for backward-compat (unused directly)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--val-interval", type=int, default=2000)  # kept for compatibility; validation is per-epoch here
    ap.add_argument("--val-max-samples", type=int, default=-1)
    ap.add_argument("--edge-mode", type=str, default="or", choices=["or", "and", "directed"])

    # ---------- throughput knobs ----------
    ap.add_argument("--batch-size", type=int, default=20, help="graphs per batch (concat-batched).")
    ap.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 0), help="DataLoader workers.")
    ap.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per worker.")
    ap.add_argument("--pin-memory", type=lambda x: str(x).lower() != "false", default=True,
                    help="Use pinned memory for faster H2D copies (True/False).")
    ap.add_argument("--persistent-workers", type=lambda x: str(x).lower() != "false", default=True,
                    help="Keep workers alive between epochs (True/False).")

    # ---------- 2-stage training knobs ----------
    ap.add_argument("--pretrain-steps", type=int, default=200000, help="Number of optimizer steps on mixed_train_set.")
    ap.add_argument("--ft-epochs", type=int, default=80, help="Fine-tune epochs on cubic_train_set.")
    ap.add_argument("--pre-lr", type=float, default=1e-3, help="LR during pretraining.")
    ap.add_argument("--ft-lr", type=float, default=5e-4, help="LR during finetuning.")

    ap.add_argument("--resume-from", type=str, default="", help="Path to checkpoint to resume weights from.")
    ap.add_argument("--strict-resume", type=lambda x: str(x).lower() != "false", default=True,
                    help="Strictly load optimizer/scheduler states if available.")

    # ---------- NEW: gradient accumulation & clipping ----------
    ap.add_argument("--accum-steps", type=int, default=4,
                    help="Accumulate gradients over this many micro-batches before optimizer.step().")
    ap.add_argument("--clip-grad-norm", type=float, default=0.0,
                    help="If >0, apply torch.nn.utils.clip_grad_norm_ with this max norm.")

    return ap.parse_args()

def main():
    args = parse_args()
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size <= 1:
        print("[WARN] Running a single process (world_size=1).")
        train_worker(0, 1, args)
    else:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

