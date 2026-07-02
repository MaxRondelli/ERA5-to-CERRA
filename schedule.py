import math
import torch


class DiffusionSchedule:
    """Base: diffusion_times ∈ [0, 1] → (signal_rates, noise_rates)."""

    def __call__(self, diffusion_times: torch.Tensor):
        noise_powers = self.get_noise_powers(diffusion_times)
        noise_rates  = torch.sqrt(noise_powers)
        signal_rates = torch.sqrt(1.0 - noise_powers)
        return signal_rates, noise_rates

    def get_noise_powers(self, diffusion_times: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CosineSchedule(DiffusionSchedule):
    """Cosine schedule parameterised by min/max signal rates (paper default)."""

    def __init__(self, min_signal_rate: float = 0.015, max_signal_rate: float = 0.95):
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate

    def get_noise_powers(self, diffusion_times: torch.Tensor) -> torch.Tensor:
        start_angle = math.acos(self.max_signal_rate)
        end_angle   = math.acos(self.min_signal_rate)
        angles      = start_angle + diffusion_times * (end_angle - start_angle)
        return torch.sin(angles) ** 2


class LinearSchedule(DiffusionSchedule):
    def __init__(self, start_noise_power: float = 1e-4, end_noise_power: float = 0.9999):
        self.start_noise_power = start_noise_power
        self.end_noise_power   = end_noise_power

    def get_noise_powers(self, diffusion_times: torch.Tensor) -> torch.Tensor:
        return self.start_noise_power + diffusion_times * (
            self.end_noise_power - self.start_noise_power
        )
