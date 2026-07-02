"""
PyTorch Dataset classes for ERA5→CERRA training.

ERA5 is stored as a numpy memmap at its native (~52×52) resolution.
In __getitem__ it is bilinear-upscaled to image_size × image_size so that
the U-Net conditioning matches the CERRA spatial resolution.

Both datasets and data loaders are provided:
  - ERA5CERRADataset  : upscales ERA5 to (image_size, image_size)
  - ERA5CERRALowRes   : keeps ERA5 at native resolution (for ESPCN / SR baselines)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data.checks import assert_cerra_era5_aligned
from setup import image_size, num_frames, max_high_res, max_low_res


class ERA5CERRADataset(Dataset):
    """
    Returns (conditioning, target):
      conditioning : (num_frames, image_size, image_size) float32  — ERA5 bilinear-upscaled
      target       : (1,          image_size, image_size) float32  — CERRA
    Both are max-normalised to [0, 1].
    """

    def __init__(
        self,
        era5_path:        str,
        cerra_path:       str,
        dataset_length:   int,
        era5_h:           int  = 52,
        era5_w:           int  = 52,
        shuffle:          bool = True,
    ):
        self.era5  = np.memmap(era5_path,  dtype="float32", mode="r",
                               shape=(dataset_length, era5_h, era5_w))
        self.cerra = np.memmap(cerra_path, dtype="float32", mode="r",
                               shape=(dataset_length, image_size, image_size))
        self.dataset_length = dataset_length
        self.era5_h         = era5_h
        self.era5_w         = era5_w

        # Valid indices: need at least num_frames prior timesteps
        self.indices = np.arange(num_frames - 1, dataset_length)
        if shuffle:
            np.random.shuffle(self.indices)

        _split = cerra_path.replace("\\", "/").split("/")[-1][len("cerra_"):-len(".npy")]
        assert_cerra_era5_aligned(self.cerra, self.era5, split=_split)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = int(self.indices[idx])

        # (num_frames, era5_h, era5_w) → (1, num_frames, era5_h, era5_w) for F.interpolate
        frames = torch.from_numpy(
            self.era5[t - num_frames + 1 : t + 1].copy()
        ).unsqueeze(0)                                       # (1, F, H_low, W_low)

        # Bilinear upsample each frame to image_size
        frames_up = F.interpolate(
            frames, size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)                                         # (num_frames, H, W)

        conditioning = frames_up / max_low_res               # normalise
        target       = torch.from_numpy(
            self.cerra[t].copy()
        ).unsqueeze(0) / max_high_res                        # (1, H, W)

        return conditioning, target


class ERA5CERRALowResDataset(Dataset):
    """
    Like ERA5CERRADataset but ERA5 is NOT upscaled (kept at era5_h × era5_w).
    Useful for non-conditioned super-resolution baselines.

    Returns:
      era5_lr  : (num_frames, era5_h, era5_w)
      cerra_hr : (1, image_size, image_size)
    """

    def __init__(
        self,
        era5_path:        str,
        cerra_path:       str,
        dataset_length:   int,
        era5_h:           int  = 52,
        era5_w:           int  = 52,
        shuffle:          bool = True,
    ):
        self.era5  = np.memmap(era5_path,  dtype="float32", mode="r",
                               shape=(dataset_length, era5_h, era5_w))
        self.cerra = np.memmap(cerra_path, dtype="float32", mode="r",
                               shape=(dataset_length, image_size, image_size))
        self.dataset_length = dataset_length
        self.indices        = np.arange(num_frames - 1, dataset_length)
        if shuffle:
            np.random.shuffle(self.indices)

        _split = cerra_path.replace("\\", "/").split("/")[-1][len("cerra_"):-len(".npy")]
        assert_cerra_era5_aligned(self.cerra, self.era5, split=_split)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t         = int(self.indices[idx])
        era5_lr   = torch.from_numpy(
            self.era5[t - num_frames + 1 : t + 1].copy()
        ) / max_low_res
        cerra_hr  = torch.from_numpy(
            self.cerra[t].copy()
        ).unsqueeze(0) / max_high_res
        return era5_lr, cerra_hr


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_dataloader(
    era5_path:      str,
    cerra_path:     str,
    dataset_length: int,
    batch_size:     int  = 32,
    era5_h:         int  = 52,
    era5_w:         int  = 52,
    shuffle:        bool = True,
    num_workers:    int  = 4,
    low_res:        bool = False,
) -> DataLoader:
    Cls = ERA5CERRALowResDataset if low_res else ERA5CERRADataset
    ds  = Cls(
        era5_path      = era5_path,
        cerra_path     = cerra_path,
        dataset_length = dataset_length,
        era5_h         = era5_h,
        era5_w         = era5_w,
        shuffle        = shuffle,
    )
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,   # Dataset-level shuffle already done
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
