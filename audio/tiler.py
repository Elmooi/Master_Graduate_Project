from __future__ import annotations  # 3.9 compat (TTS env)

import math
import torch
import torch.nn as nn


class WaveformTiler(nn.Module):
    """
    Waveform-domain UAP tiler.

    Tiles a fixed-size delta patch over an arbitrary-length waveform by
    repeating and cropping, then applies additive perturbation with optional
    per-sample SNR normalization.

    No STFT / iSTFT involved — fully differentiable in the time domain.
    """

    def __init__(
        self,
        noise_level: float = 0.05,
        target_snr_db: float | None = None,
    ):
        super().__init__()
        self.noise_level = noise_level
        self.target_snr_db = target_snr_db

    def forward(
        self,
        waveform: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            waveform: [B, N]  clean audio batch
            delta:    [B, T_patch]  UAP waveform patch

        Returns:
            protected: [B, N]
        """
        B, N = waveform.shape
        T_patch = delta.shape[-1]

        # Tile delta to cover N samples
        repeats = math.ceil(N / T_patch)
        delta_tiled = delta.repeat(1, repeats)[:, :N]   # [B, N]

        protected = waveform + delta_tiled * self.noise_level

        if self.target_snr_db is not None:
            # Forced fp32: under autocast, 1e-8 (the clamp floor below)
            # rounds to exactly 0.0 in fp16 (torch.finfo(float16).tiny is
            # ~6e-5, and 1e-8 is well below even the smallest subnormal),
            # so the clamp is a no-op and noise_power/sig_power can
            # genuinely divide-by-zero -> inf -> nan once multiplied back
            # through `scale`. Confirmed via diag_overfit_one_speaker.py:
            # this NaN'd every single step regardless of GradScaler's
            # scale (down to scale=1), which rules out fp16 dynamic-range
            # overflow and points at exact-zero underflow instead — this
            # is that bug.
            with torch.autocast(device_type=waveform.device.type, enabled=False):
                noise = (protected - waveform).float()
                waveform_f = waveform.float()
                sig_power = waveform_f.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
                noise_power = noise.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
                target_noise_power = sig_power / (10 ** (self.target_snr_db / 10))
                scale = (target_noise_power / noise_power).sqrt()
                protected = (waveform_f + noise * scale).to(waveform.dtype)

        return protected.clamp(-1.0, 1.0)
