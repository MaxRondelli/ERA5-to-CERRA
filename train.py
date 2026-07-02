"""
Training script for the ERA5→CERRA diffusion downscaling model.

Usage
-----
    python train.py
    python train.py --epochs 200 --batch_size 32 --device cuda
    python train.py --resume checkpoints/epoch_050.pt

The script:
  1. Builds the denoising U-Net and the DiffusionModel wrapper.
  2. Computes mean/variance statistics on the training set for normalisation.
  3. Runs the training loop; saves a checkpoint every 10 epochs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from setup import (
    image_size, num_frames, widths, block_depth,
    embedding_dims, embedding_max_frequency,
    min_signal_rate, max_signal_rate,
    ema, learning_rate, weight_decay, batch_size, num_epochs,
    dataset_length_2010_2019,
)
from denoising_unet import get_network
from diffusion_model import DiffusionModel
from schedule import CosineSchedule
from generators import make_dataloader


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(memmap_path: str, N: int, shape: tuple, n_samples: int = 5000) -> tuple:
    """Estimate mean and variance by sampling n_samples entries."""
    mm  = np.memmap(memmap_path, dtype="float32", mode="r", shape=(N, *shape))
    idx = np.random.choice(N, min(n_samples, N), replace=False)
    sub = mm[idx].astype(np.float64)
    return float(sub.mean()), float(sub.var())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--ckpt_dir",   default="checkpoints")
    parser.add_argument("--epochs",     type=int,   default=num_epochs)
    parser.add_argument("--batch_size", type=int,   default=batch_size)
    parser.add_argument("--lr",         type=float, default=learning_rate)
    parser.add_argument("--wd",         type=float, default=weight_decay)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    args = parser.parse_args()

    device   = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ── ERA5 spatial size ──────────────────────────────────────────────────────
    era5_shape = np.load(data_dir / "era5_shape.npy")
    era5_h, era5_w = int(era5_shape[0]), int(era5_shape[1])
    print(f"ERA5 native size: {era5_h}×{era5_w}")

    # ── Dataset statistics ─────────────────────────────────────────────────────
    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print("[stats] loaded from cache")
    else:
        print("[stats] computing …")
        mean_cond, var_cond = compute_stats(
            str(data_dir / "era5_train.npy"),
            dataset_length_2010_2019,
            (era5_h, era5_w),
        )
        # Generator divides by max_low_res first, so stats are on [0,1]-range data
        from setup import max_low_res, max_high_res
        mean_cond  /= max_low_res
        var_cond   /= max_low_res ** 2
        mean_tgt, var_tgt = compute_stats(
            str(data_dir / "cerra_train.npy"),
            dataset_length_2010_2019,
            (image_size, image_size),
        )
        mean_tgt  /= max_high_res
        var_tgt   /= max_high_res ** 2
        stats = {
            "mean_conditioning": mean_cond,   "variance_conditioning": var_cond,
            "mean_target":       mean_tgt,    "variance_target":       var_tgt,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[stats] saved to {stats_path}")

    print("[stats]", stats)

    # ── Data loaders ───────────────────────────────────────────────────────────
    train_loader = make_dataloader(
        era5_path      = str(data_dir / "era5_train.npy"),
        cerra_path     = str(data_dir / "cerra_train.npy"),
        dataset_length = dataset_length_2010_2019,
        batch_size     = args.batch_size,
        era5_h         = era5_h,
        era5_w         = era5_w,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # input_channels = num_frames (ERA5) + 1 (noisy CERRA)
    network = get_network(
        input_channels          = num_frames + 1,
        output_channels         = 1,
        widths                  = widths,
        block_depth             = block_depth,
        embedding_dims          = embedding_dims,
        embedding_max_frequency = embedding_max_frequency,
        use_cbam_bottleneck     = True,
    ).to(device)

    schedule = CosineSchedule(min_signal_rate, max_signal_rate)

    model = DiffusionModel(
        network               = network,
        diffusion_schedule    = schedule,
        image_size_h          = image_size,
        image_size_w          = image_size,
        ema                   = ema,
        prediction_type       = "velocity",
        loss_type             = "velocity",
        mean_target           = stats["mean_target"],
        variance_target       = stats["variance_target"],
        mean_conditioning     = stats["mean_conditioning"],
        variance_conditioning = stats["variance_conditioning"],
        output_frames         = 1,
        clip_pred_images      = True,
    ).to(device)

    optimizer = optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.wd)
    start_epoch = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.network.load_state_dict(ckpt["network"])
        model.ema_network.load_state_dict(ckpt["ema_network"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] epoch {start_epoch}")

    # ── Training loop ──────────────────────────────────────────────────────────
    model.train()
    loader_iter = iter(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_metrics = {"loss": 0, "i_loss": 0, "n_loss": 0, "v_loss": 0}
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            try:
                conditioning, target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                conditioning, target = next(loader_iter)

            conditioning = conditioning.to(device)
            target       = target.to(device)

            metrics = model.train_step(conditioning, target, optimizer)
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]

        elapsed = time.time() - t0
        avg     = {k: v / args.steps_per_epoch for k, v in epoch_metrics.items()}
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"{elapsed:.0f}s  "
            f"i_loss={avg['i_loss']:.4f}  "
            f"n_loss={avg['n_loss']:.4f}  "
            f"v_loss={avg['v_loss']:.4f}"
        )

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch":       epoch,
                    "network":     model.network.state_dict(),
                    "ema_network": model.ema_network.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "stats":       stats,
                },
                ckpt_path,
            )
            print(f"[ckpt]  saved {ckpt_path}")


if __name__ == "__main__":
    main()
