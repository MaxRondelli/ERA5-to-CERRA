"""
Evaluate a trained ERA5→CERRA diffusion model on the test split.

Usage
-----
    python evaluate.py --checkpoint checkpoints/epoch_200.pt
    python evaluate.py --checkpoint checkpoints/epoch_200.pt \\
                       --diffusion_steps 20 --n_samples 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup import (
    image_size, num_frames, widths, block_depth,
    embedding_dims, embedding_max_frequency,
    min_signal_rate, max_signal_rate,
    ema, batch_size,
    dataset_length_2009, dataset_length_2020,
    max_high_res, max_low_res,
)
from denoising_unet import get_network
from diffusion_model import DiffusionModel
from schedule import CosineSchedule
from generators import make_dataloader
from utils import batch_ssim, batch_psnr


def load_model(checkpoint_path: str, device: torch.device) -> DiffusionModel:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    stats = ckpt["stats"]

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
        mean_target           = stats["mean_target"],
        variance_target       = stats["variance_target"],
        mean_conditioning     = stats["mean_conditioning"],
        variance_conditioning = stats["variance_conditioning"],
        output_frames         = 1,
    ).to(device)

    model.network.load_state_dict(ckpt["network"])
    model.ema_network.load_state_dict(ckpt["ema_network"])
    model.eval()
    return model

def evaluate(args):
    device   = torch.device(args.device)
    data_dir = Path(args.data_dir)

    era5_shape = np.load(data_dir / "era5_shape.npy")
    era5_h, era5_w = int(era5_shape[0]), int(era5_shape[1])

    split_len = dataset_length_2020 if args.split == "test" else dataset_length_2009
    loader    = make_dataloader(
        era5_path      = str(data_dir / f"era5_{args.split}.npy"),
        cerra_path     = str(data_dir / f"cerra_{args.split}.npy"),
        dataset_length = split_len,
        batch_size     = args.batch_size,
        era5_h         = era5_h,
        era5_w         = era5_w,
        shuffle        = False,
    )

    model = load_model(args.checkpoint, device)

    ssim_scores, psnr_scores = [], []
    samples_done = 0

    for conditioning, target in loader:
        if samples_done >= args.n_samples:
            break
        conditioning = conditioning.to(device)
        target_np    = target.numpy()                    # (B, 1, H, W) [0,1]

        pred_norm = model.generate(conditioning, args.diffusion_steps)
        pred_np   = pred_norm.cpu().numpy()              # (B, 1, H, W) [0,1]

        # Convert to (B, H, W, 1) for skimage metrics
        pred_hw  = pred_np.transpose(0, 2, 3, 1).clip(0, 1)
        tgt_hw   = target_np.transpose(0, 2, 3, 1)

        ssim_scores.append(batch_ssim(tgt_hw, pred_hw))
        psnr_scores.append(batch_psnr(tgt_hw, pred_hw))
        samples_done += conditioning.shape[0]

    mean_ssim = float(np.mean(ssim_scores))
    mean_psnr = float(np.mean(psnr_scores))
    print(f"Split={args.split}  n={samples_done}  "
          f"SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f} dB")

    # ── Save a visual comparison ───────────────────────────────────────────────
    if args.save_fig:
        conditioning, target = next(iter(loader))
        conditioning = conditioning[:1].to(device)
        pred_norm    = model.generate(conditioning, args.diffusion_steps)

        pred_ms  = pred_norm[0, 0].cpu().numpy() * max_high_res
        tgt_ms   = target[0, 0].numpy()          * max_high_res
        # Show the last ERA5 conditioning frame
        cond_ms  = conditioning[0, -1].cpu().numpy() * max_low_res

        vmin, vmax = 0, max(pred_ms.max(), tgt_ms.max(), cond_ms.max())
        fig, axes  = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(
            axes,
            [cond_ms, tgt_ms, pred_ms],
            ["ERA5 (input)", "CERRA (target)", "Diffusion output"],
        ):
            im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(title)
            ax.axis("off")
        fig.colorbar(im, ax=axes, label="Wind speed [m/s]", shrink=0.7)
        fig.savefig(args.save_fig, bbox_inches="tight", dpi=150)
        print(f"[fig] saved to {args.save_fig}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--data_dir",         default="data/processed")
    parser.add_argument("--split",            default="test", choices=["val", "test"])
    parser.add_argument("--diffusion_steps",  type=int,   default=20)
    parser.add_argument("--n_samples",        type=int,   default=200)
    parser.add_argument("--batch_size",       type=int,   default=4)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_fig",         default="eval_sample.png")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
