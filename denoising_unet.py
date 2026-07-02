"""
U-Net denoiser for the DDIM wind downscaling model (PyTorch).

Architecture (paper version):
- Sinusoidal noise-variance embedding broadcast to spatial dims
- 3 encoder stages: depthwise-separable residual blocks + AvgPool2d
- Bottleneck with CBAM (channel + spatial) attention
- 3 decoder stages: bilinear Upsample + skip connections
- Final 1×1 Conv with zero-initialised weights

Tensors are in NCHW format throughout (B, C, H, W).
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Depthwise separable convolution ───────────────────────────────────────────

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch,  kernel_size, padding=padding, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ── Sinusoidal noise embedding ─────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """
    Input : noise_variances (B, 1, 1, 1) — the squared noise rate
    Output: (B, embedding_dims, 1, 1)
    """

    def __init__(self, embedding_dims: int = 64, embedding_max_frequency: float = 1000.0):
        super().__init__()
        half        = embedding_dims // 2
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(embedding_max_frequency), half)
        )
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies)
        self.embedding_dims = embedding_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat   = x.view(-1, 1)                                    # (B, 1)
        sin_part = torch.sin(self.angular_speeds * x_flat)          # (B, half)
        cos_part = torch.cos(self.angular_speeds * x_flat)          # (B, half)
        emb      = torch.cat([sin_part, cos_part], dim=-1)          # (B, embedding_dims)
        return emb.view(-1, self.embedding_dims, 1, 1)              # (B, C, 1, 1)


# ── Residual block ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Depthwise-separable residual block with GroupNorm(8) and SiLU (≡ Swish)."""

    def __init__(self, width: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.conv1 = DepthwiseSeparableConv2d(width, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = DepthwiseSeparableConv2d(width, width)
        self.proj  = nn.Conv2d(width, width, 1)
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + residual


# ── CBAM attention ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        squeeze    = max(channels // ratio, 1)
        self.mlp   = nn.Sequential(
            nn.Linear(channels, squeeze),
            nn.ReLU(),
            nn.Linear(squeeze, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg   = x.mean(dim=(2, 3))          # (B, C)
        mx    = x.amax(dim=(2, 3))          # (B, C)
        scale = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg   = x.mean(dim=1, keepdim=True)
        mx    = x.amax(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class ResidualBlockWithCBAM(nn.Module):
    """Residual block + CBAM (channel then spatial attention)."""

    def __init__(self, width: int):
        super().__init__()
        self.norm1   = nn.GroupNorm(8, width)
        self.conv1   = DepthwiseSeparableConv2d(width, width)
        self.norm2   = nn.GroupNorm(8, width)
        self.conv2   = DepthwiseSeparableConv2d(width, width)
        self.proj    = nn.Conv2d(width, width, 1)
        self.act     = nn.SiLU()
        self.ch_att  = ChannelAttention(width)
        self.sp_att  = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        x = self.ch_att(x)
        x = self.sp_att(x)
        return x + residual


# ── Encoder / decoder stage containers ────────────────────────────────────────

class DownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, block_depth: int, use_cbam: bool = False):
        super().__init__()
        Block = ResidualBlockWithCBAM if use_cbam else ResidualBlock
        # First block adjusts channels if needed
        blocks = []
        for i in range(block_depth):
            ch = in_ch if i == 0 else out_ch
            if i == 0 and in_ch != out_ch:
                # channel projection inside first block's residual path
                pass
            blocks.append(_ResBlockWithProj(in_ch if i == 0 else out_ch, out_ch, use_cbam))
        self.blocks   = nn.ModuleList(blocks)
        self.pool     = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, block_depth: int, use_cbam: bool = False):
        super().__init__()
        blocks = []
        for i in range(block_depth):
            ch_in = in_ch + skip_ch if i == 0 else out_ch
            blocks.append(_ResBlockWithProj(ch_in, out_ch, use_cbam))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x    = torch.cat([x, skip], dim=1)
            x    = block(x)
        return x


class _ResBlockWithProj(nn.Module):
    """Residual block where in_ch may differ from out_ch (handles channel changes)."""

    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = False):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, in_ch)
        self.conv1  = DepthwiseSeparableConv2d(in_ch, out_ch)
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.conv2  = DepthwiseSeparableConv2d(out_ch, out_ch)
        self.proj   = nn.Conv2d(in_ch, out_ch, 1)
        self.act    = nn.SiLU()
        self.ch_att = ChannelAttention(out_ch) if use_cbam else None
        self.sp_att = SpatialAttention()       if use_cbam else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        if self.ch_att is not None:
            x = self.ch_att(x)
            x = self.sp_att(x)
        return x + residual


# ── Full U-Net ─────────────────────────────────────────────────────────────────

class DenoisingUNet(nn.Module):
    """
    Inputs
    ------
    noisy_images    : (B, input_channels, H, W)
                      = torch.cat([conditioning, noisy_target], dim=1)
    noise_variances : (B, 1, 1, 1)  — noise_rate²

    Output
    ------
    (B, output_channels, H, W) — predicted velocity / image / noise
    """

    def __init__(
        self,
        input_channels:          int,
        output_channels:         int,
        widths:                  list,
        block_depth:             int,
        embedding_dims:          int   = 64,
        embedding_max_frequency: float = 1000.0,
        use_cbam_bottleneck:     bool  = True,
    ):
        super().__init__()
        assert len(widths) == 4, "widths must have 4 elements"

        self.noise_emb = SinusoidalEmbedding(embedding_dims, embedding_max_frequency)

        # Project image + embedding to first width
        self.input_proj = nn.Conv2d(input_channels + embedding_dims, widths[0], 1)

        # Encoder (3 stages, depth-wise resolution halving)
        self.enc0 = _make_down_blocks(widths[0], widths[0], block_depth)
        self.enc1 = _make_down_blocks(widths[0], widths[1], block_depth)
        self.enc2 = _make_down_blocks(widths[1], widths[2], block_depth)
        self.pool  = nn.AvgPool2d(2)

        # Bottleneck
        BottleneckBlock = ResidualBlockWithCBAM if use_cbam_bottleneck else ResidualBlock
        self.bottleneck = nn.ModuleList(
            [_ResBlockWithProj(widths[2] if i == 0 else widths[3], widths[3], use_cbam_bottleneck)
             for i in range(block_depth)]
        )

        # Decoder (skip channels come from encoder stages)
        self.dec2 = _make_up_blocks(widths[3], widths[2], widths[2], block_depth)
        self.dec1 = _make_up_blocks(widths[2], widths[1], widths[1], block_depth)
        self.dec0 = _make_up_blocks(widths[1], widths[0], widths[0], block_depth)

        # Final projection — zeros init so the model starts from identity-like state
        self.output_proj = nn.Conv2d(widths[0], output_channels, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_images:    torch.Tensor,
        noise_variances: torch.Tensor,
    ) -> torch.Tensor:
        H, W = noisy_images.shape[2], noisy_images.shape[3]

        # Sinusoidal embedding broadcast to spatial dims
        emb = self.noise_emb(noise_variances)                       # (B, emb_dims, 1, 1)
        emb = emb.expand(-1, -1, H, W)                             # (B, emb_dims, H, W)

        x = self.input_proj(torch.cat([noisy_images, emb], dim=1)) # (B, w0, H, W)

        skips = []

        # Encoder
        x, s0 = _forward_enc(self.enc0, x);  skips.extend(s0);  x = self.pool(x)
        x, s1 = _forward_enc(self.enc1, x);  skips.extend(s1);  x = self.pool(x)
        x, s2 = _forward_enc(self.enc2, x);  skips.extend(s2);  x = self.pool(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        # Decoder
        x = _forward_dec(self.dec2, x, skips, block_depth=len(self.dec2))
        x = _forward_dec(self.dec1, x, skips, block_depth=len(self.dec1))
        x = _forward_dec(self.dec0, x, skips, block_depth=len(self.dec0))

        return self.output_proj(x)


# ── Helpers to build stage block lists ────────────────────────────────────────

def _make_down_blocks(in_ch: int, out_ch: int, depth: int) -> nn.ModuleList:
    blocks = []
    for i in range(depth):
        ci = in_ch if i == 0 else out_ch
        blocks.append(_ResBlockWithProj(ci, out_ch))
    return nn.ModuleList(blocks)


def _make_up_blocks(in_ch: int, skip_ch: int, out_ch: int, depth: int) -> nn.ModuleList:
    blocks = []
    for i in range(depth):
        ci = (in_ch if i == 0 else out_ch) + skip_ch
        blocks.append(_ResBlockWithProj(ci, out_ch))
    return nn.ModuleList(blocks)


def _forward_enc(blocks: nn.ModuleList, x: torch.Tensor):
    skips = []
    for block in blocks:
        x = block(x)
        skips.append(x)
    return x, skips


def _forward_dec(
    blocks:      nn.ModuleList,
    x:           torch.Tensor,
    skips:       list,
    block_depth: int,
) -> torch.Tensor:
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    for block in blocks:
        skip = skips.pop()
        x    = torch.cat([x, skip], dim=1)
        x    = block(x)
    return x


# ── Convenience builder matching the paper's defaults ─────────────────────────

def get_network(
    input_channels:          int,
    output_channels:         int,
    widths:                  list,
    block_depth:             int,
    embedding_dims:          int   = 64,
    embedding_max_frequency: float = 1000.0,
    use_cbam_bottleneck:     bool  = True,
) -> DenoisingUNet:
    return DenoisingUNet(
        input_channels          = input_channels,
        output_channels         = output_channels,
        widths                  = widths,
        block_depth             = block_depth,
        embedding_dims          = embedding_dims,
        embedding_max_frequency = embedding_max_frequency,
        use_cbam_bottleneck     = use_cbam_bottleneck,
    )
