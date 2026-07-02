"""
DDIM diffusion model (PyTorch).

Training
--------
  1. Normalise conditioning and target with per-dataset mean/variance.
  2. Mix target with Gaussian noise at a random diffusion time.
  3. Predict velocity (or image / noise) with the denoising U-Net.
  4. Compute loss; update network and EMA network.

Inference
---------
  reverse_diffusion(): DDIM iterative denoising.
  generate(): convenience wrapper — normalises input, runs reverse diffusion,
              denormalises output back to [0, 1].
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(
        self,
        network,
        diffusion_schedule,
        image_size_h:         int,
        image_size_w:         int,
        ema:                  float = 0.999,
        prediction_type:      str   = "velocity",
        loss_type:            str   = "velocity",
        mean_target:          float = 0.0,
        variance_target:      float = 1.0,
        mean_conditioning:    float = 0.0,
        variance_conditioning:float = 1.0,
        output_frames:        int   = 1,
        clip_pred_images:     bool  = False,
    ):
        super().__init__()
        self.network              = network
        self.ema_network          = copy.deepcopy(network)
        for p in self.ema_network.parameters():
            p.requires_grad_(False)

        self.diffusion_schedule   = diffusion_schedule
        self.image_size_h         = image_size_h
        self.image_size_w         = image_size_w
        self.ema                  = ema
        self.prediction_type      = prediction_type
        self.loss_type            = loss_type
        self.mean_target          = mean_target
        self.variance_target      = variance_target
        self.mean_conditioning    = mean_conditioning
        self.variance_conditioning = variance_conditioning
        self.output_frames        = output_frames
        self.clip_pred_images     = clip_pred_images

    # ── Normalisation ──────────────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor, mean: float, variance: float) -> torch.Tensor:
        return (x - mean) / variance ** 0.5

    def denormalize(self, x: torch.Tensor, mean: float, variance: float) -> torch.Tensor:
        return x * variance ** 0.5 + mean

    # ── Decompose network output ───────────────────────────────────────────────

    def get_components(self, noisy_images, predictions, signal_rates, noise_rates):
        """
        Recover pred_velocities, pred_images, pred_noises from the raw network output.

        With signal² + noise² = 1:
          noisy = signal * target + noise * ε
          v     = signal * ε    − noise * target
          ⇒ pred_image = signal * noisy − noise * v
          ⇒ pred_noise = noise  * noisy + signal * v
        """
        if self.prediction_type == "velocity":
            pred_images     = signal_rates * noisy_images - noise_rates  * predictions
            pred_noises     = noise_rates  * noisy_images + signal_rates * predictions
            pred_velocities = predictions
        elif self.prediction_type == "signal":
            pred_images     = predictions
            pred_noises     = (noisy_images - signal_rates * pred_images) / noise_rates
            pred_velocities = signal_rates * pred_noises - noise_rates * pred_images
        elif self.prediction_type == "noise":
            pred_noises     = predictions
            pred_images     = (noisy_images - noise_rates * pred_noises) / signal_rates
            pred_velocities = signal_rates * pred_noises - noise_rates * pred_images
        else:
            raise NotImplementedError(self.prediction_type)

        if self.clip_pred_images:
            pred_images = pred_images.clamp(-3.0, 3.0)

        return pred_velocities, pred_images, pred_noises

    # ── Training step ──────────────────────────────────────────────────────────

    def train_step(
        self,
        conditioning: torch.Tensor,   # (B, num_frames, H, W)
        target:       torch.Tensor,   # (B, 1, H, W)
        optimizer:    torch.optim.Optimizer,
    ) -> dict:
        device = next(self.network.parameters()).device
        B      = conditioning.shape[0]

        target       = self.normalize(target,       self.mean_target,       self.variance_target)
        conditioning = self.normalize(conditioning, self.mean_conditioning, self.variance_conditioning)

        noises = torch.randn(
            B, self.output_frames, self.image_size_h, self.image_size_w, device=device
        )
        diffusion_times = torch.rand(B, 1, 1, 1, device=device)
        signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * target + noise_rates * noises
        velocities   = signal_rates * noises - noise_rates * target

        net_input = torch.cat([conditioning, noisy_images], dim=1)
        optimizer.zero_grad()
        predictions = self.network(net_input, noise_rates ** 2)

        pred_velocities, pred_images, pred_noises = self.get_components(
            noisy_images, predictions, signal_rates, noise_rates
        )
        velocity_loss = F.mse_loss(pred_velocities, velocities)
        image_loss    = F.mse_loss(pred_images,     target)
        noise_loss    = F.mse_loss(pred_noises,     noises)

        loss = {"velocity": velocity_loss, "signal": image_loss, "noise": noise_loss}[
            self.loss_type
        ]
        loss.backward()
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for p, ep in zip(self.network.parameters(), self.ema_network.parameters()):
                ep.data.mul_(self.ema).add_(p.data, alpha=1.0 - self.ema)

        return {
            "loss":   loss.item(),
            "i_loss": image_loss.item(),
            "n_loss": noise_loss.item(),
            "v_loss": velocity_loss.item(),
        }

    # ── Reverse diffusion (DDIM) ───────────────────────────────────────────────

    @torch.no_grad()
    def reverse_diffusion(
        self,
        conditioning:   torch.Tensor,   # (B, num_frames, H, W) — already normalised
        initial_noise:  torch.Tensor,   # (B, 1, H, W)
        diffusion_steps: int,
    ) -> torch.Tensor:
        device       = conditioning.device
        B            = conditioning.shape[0]
        step_size    = 1.0 / diffusion_steps
        noisy_images = initial_noise

        for step in range(diffusion_steps):
            t = torch.ones(B, 1, 1, 1, device=device) - step * step_size
            signal_rates, noise_rates = self.diffusion_schedule(t)

            net_input   = torch.cat([conditioning, noisy_images], dim=1)
            predictions = self.ema_network(net_input, noise_rates ** 2)

            _, pred_images, pred_noises = self.get_components(
                noisy_images, predictions, signal_rates, noise_rates
            )
            next_sr, next_nr = self.diffusion_schedule(t - step_size)
            noisy_images     = next_sr * pred_images + next_nr * pred_noises

        return pred_images

    # ── Public inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        conditioning:    torch.Tensor,
        diffusion_steps: int,
        seed:            int | None = None,
    ) -> torch.Tensor:
        """
        conditioning : (B, num_frames, H, W) max-normalised [0, 1]
        Returns      : (B, 1, H, W) max-normalised [0, 1]
        """
        device = conditioning.device
        if seed is not None:
            torch.manual_seed(seed)

        cond_norm = self.normalize(
            conditioning, self.mean_conditioning, self.variance_conditioning
        )
        initial_noise = torch.randn(
            conditioning.shape[0], self.output_frames,
            self.image_size_h, self.image_size_w,
            device=device,
        )
        pred = self.reverse_diffusion(cond_norm, initial_noise, diffusion_steps)
        return self.denormalize(pred, self.mean_target, self.variance_target)
