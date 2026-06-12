"""Training loop for DarkVesselNet.

Engineering choices that matter at this data scale:
  * mixed precision (autocast + GradScaler) — halves memory, speeds up
    training on any modern GPU;
  * gradient accumulation — lets you simulate a bigger batch when GPU
    memory is tight;
  * positive-aware sampling — see data/dataset.py;
  * a quick chip-level detection F1 on the validation set every epoch, so
    you watch the metric you care about, not just the loss.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import ChipDataset, PositiveAwareSampler
from .inference.peaks import extract_peaks
from .models.losses import MultiTaskLoss
from .models.model import DarkVesselNet


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def chip_f1(pred_peaks: np.ndarray, gt_rows: np.ndarray, gt_cols: np.ndarray,
            radius_px: float = 20.0) -> tuple[int, int, int]:
    """Greedy match of predicted peaks to ground-truth points on one chip.

    Returns (true positives, false positives, false negatives).
    radius_px=20 corresponds to the xView3 200 m matching radius at 10 m/px.
    """
    n_gt = len(gt_rows)
    if len(pred_peaks) == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, len(pred_peaks), 0
    used = np.zeros(n_gt, dtype=bool)
    tp = 0
    for r, c, _ in pred_peaks:
        d = np.hypot(gt_rows - r, gt_cols - c)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius_px:
            used[j] = True
            tp += 1
    return tp, len(pred_peaks) - tp, n_gt - tp


def evaluate(model, loader, device, score_threshold: float) -> dict:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            out = model(images)
            heat = torch.sigmoid(out["heatmap"])
            for i in range(images.shape[0]):
                peaks = extract_peaks(heat[i, 0].cpu().numpy(), score_threshold)
                valid = batch["valid"][i].bool()
                gt_r = batch["pt_rows"][i][valid].numpy().astype(float)
                gt_c = batch["pt_cols"][i][valid].numpy().astype(float)
                a, b, c = chip_f1(peaks, gt_r, gt_c)
                tp, fp, fn = tp + a, fp + b, fn + c
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def train(cfg, train_index: str, val_index: str, resume_from: str | None = None):
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = pick_device()
    print(f"Training on device: {device}")

    train_ds = ChipDataset(train_index, cfg, augment=True)
    val_ds = ChipDataset(val_index, cfg, augment=False)

    # The per-epoch validation is a progress signal, not the final verdict
    # (that's scripts 04/05 on full scenes) — so cap it at a fixed random
    # subset of chips to keep epochs fast on big validation splits.
    max_val = cfg.train.get("val_max_chips", 400)
    n_val_total = len(val_ds.index)
    if n_val_total > max_val:
        val_ds.index = (val_ds.index.sample(n=max_val, random_state=cfg.train.seed)
                        .reset_index(drop=True))
        print(f"Validation capped at {max_val} random chips (of {n_val_total})")

    sampler = PositiveAwareSampler(train_ds.index, cfg.train.positive_fraction,
                                   seed=cfg.train.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              sampler=sampler, num_workers=cfg.train.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers)

    model = DarkVesselNet(cfg).to(device)
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed weights from {resume_from} "
              f"(epoch {ckpt.get('epoch', '?')}, "
              f"F1 {ckpt.get('metrics', {}).get('f1', float('nan')):.3f})")
    criterion = MultiTaskLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs)
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0 = time.time()
        running = {}
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch["image"])
                losses = criterion(outputs, batch)
                loss = losses["total"] / cfg.train.grad_accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % cfg.train.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())

        scheduler.step()
        n_batches = max(len(train_loader), 1)
        loss_str = "  ".join(f"{k}={v / n_batches:.4f}" for k, v in running.items())
        print(f"epoch {epoch}  [{time.time() - t0:.0f}s]  {loss_str}")

        if epoch % cfg.train.val_every == 0:
            metrics = evaluate(model, val_loader, device,
                               cfg.inference.score_threshold)
            print(f"  val: P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
            cfg_plain = cfg.to_plain() if hasattr(cfg, "to_plain") else dict(cfg)
            torch.save({"model": model.state_dict(), "cfg": cfg_plain,
                        "epoch": epoch, "metrics": metrics},
                       ckpt_dir / "last.pt")
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save({"model": model.state_dict(), "cfg": cfg_plain,
                            "epoch": epoch, "metrics": metrics},
                           ckpt_dir / "best.pt")
                print(f"  new best F1 ({best_f1:.3f}) -> saved best.pt")

    print(f"Done. Best validation F1: {best_f1:.3f}")
