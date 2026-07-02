# ── Dataset statistics (ERA5 & CERRA, 2010-2019 training set) ─────────────────
max_high_res = 31.347172   # CERRA max wind speed [m/s]
max_low_res  = 26.298004   # ERA5  max wind speed [m/s]

dataset_length_2010_2019 = 29216
dataset_length_2020      = 2928
dataset_length_2009      = 2920

# ── Spatial / temporal dims ────────────────────────────────────────────────────
image_size = 256       # CERRA crop (pixels, square)
num_frames = 4         # consecutive ERA5 frames used as conditioning

# ── Inference ─────────────────────────────────────────────────────────────────
plot_diffusion_steps = 20

# ── Noise schedule ─────────────────────────────────────────────────────────────
min_signal_rate = 0.015
max_signal_rate = 0.95

# ── Sinusoidal noise embedding ─────────────────────────────────────────────────
embedding_dims          = 64
embedding_max_frequency = 1000.0

# ── U-Net architecture ─────────────────────────────────────────────────────────
widths      = [64, 128, 256, 384]   # channels at each resolution stage
block_depth = 3                      # residual blocks per stage

# ── Optimiser ─────────────────────────────────────────────────────────────────
ema           = 0.999
learning_rate = 1e-3
weight_decay  = 1e-4
batch_size    = 8
num_epochs    = 200    # >= 50 gives acceptable results
