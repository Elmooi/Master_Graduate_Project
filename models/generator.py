import math

import torch
import torch.nn as nn


class ResUpBlock1d(nn.Module):
    """2x upsample + residual conv block (WaveGAN-style). InstanceNorm, not
    BatchNorm — the generator gets called with batch size as low as 1
    (single-query episodes) during training, which BatchNorm can't handle."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.InstanceNorm1d(out_ch, affine=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.InstanceNorm1d(out_ch, affine=True)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.act = nn.GELU()
        # Plain addition of h+skip at every one of many stacked blocks lets
        # variance grow roughly with sqrt(depth). Fine for 8 stages
        # (patch_length=4096), but at 13 stages (patch_length=131072) this
        # produced exploding activations (delta max |value| ~54 vs mean ~0.6,
        # grad norm ~1.8M on a fresh checkpoint) and training never moved off
        # its random-init loss. This 1/sqrt(2) rescale is the standard fix
        # for residual stacks this deep (keeps output variance ≈ input
        # variance per block instead of compounding).
        self.res_scale = 1.0 / math.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        skip = self.skip(x_up)
        h = self.act(self.norm1(self.conv1(x_up)))
        h = self.norm2(self.conv2(h))
        return self.act((h + skip) * self.res_scale)


class WaveformUAPGeneratorConv(nn.Module):
    """
    WaveGAN-style waveform generator: project z_u to a short "seed" sequence,
    then progressively 2x-upsample with residual conv blocks up to
    patch_length — as opposed to WaveformUAPGenerator's single dense MLP that
    jumps straight from z_dim to patch_length in one step.

    Motivation: across every fix we tried for the CAM++-only UAP (fixing the
    2s audio truncation, fixing generator mode collapse via the relation
    loss), val_adv plateaued in the same ~0.42-0.44 band regardless — see
    train_log_campp.txt. A per-sample PGD attack on the same target (CAM++)
    at a similar noise budget (attack_campplus.py) pushes cos much lower, so
    the ceiling isn't the noise budget; it looks like the plain MLP can't
    find as sharp a direction as a directly-optimized per-sample delta can.
    Conv + residual upsampling is the standard architecture for this kind of
    "small conditioning vector -> long structured 1D signal" generation task
    (WaveGAN, DCGAN-style decoders) and gives the network more room to shape
    the signal at multiple time resolutions instead of one flat projection.

    Input : z_u  [B, z_dim]
    Output: delta [B, patch_length]
    """

    def __init__(
        self,
        z_dim: int = 192,
        patch_length: int = 4096,
        base_channels: int = 512,
        seed_len: int = 16,
    ):
        super().__init__()
        n_stages = int(math.log2(patch_length // seed_len))
        assert seed_len * (2 ** n_stages) == patch_length, (
            f"patch_length ({patch_length}) must be seed_len ({seed_len}) * a power of 2"
        )

        self.seed_len = seed_len
        self.base_channels = base_channels
        self.patch_length = patch_length

        self.proj = nn.Linear(z_dim, base_channels * seed_len)

        channels = [base_channels]
        for _ in range(n_stages):
            channels.append(max(32, channels[-1] // 2))

        self.blocks = nn.ModuleList([
            ResUpBlock1d(channels[i], channels[i + 1]) for i in range(n_stages)
        ])
        self.out_conv = nn.Conv1d(channels[-1], 1, kernel_size=7, padding=3)
        # Start near-zero instead of default-scale random weights — standard
        # practice for a generator's final layer (e.g. StyleGAN's to_rgb) so
        # training starts from a tame, small-magnitude output rather than
        # whatever a generic Kaiming/uniform init happens to produce after
        # being amplified through n_stages upsample blocks.
        nn.init.normal_(self.out_conv.weight, std=0.01)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
        Returns:
            delta: [B, patch_length]
        """
        B = z.shape[0]
        x = self.proj(z).view(B, self.base_channels, self.seed_len)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)  # [B, 1, patch_length]
        return x.squeeze(1)


class FiLM1d(nn.Module):
    """Per-channel scale/bias conditioning from z, broadcast over time.

    Zero-initialized so scale=bias=0 at init -> (1+scale)=1, bias=0 -> pure
    identity pass-through. Without this, a block-conditioned network this
    deep (8-9 stacked blocks, 2 FiLM applications each) starts by injecting
    random per-channel scale/bias noise at every block from a default-inited
    Linear, which compounds across the stack -- this was the actual cause of
    the instability/stalled loss seen vs. the plain MLP at matched param
    count (grad non-finite ~25-30% of steps vs 0%, loss flat instead of
    decreasing). Same rationale as WaveformUAPGeneratorConv's zeroed
    out_conv: start as a tame/identity function, let conditioning strength
    grow from training instead of being random from step 0.
    """

    def __init__(self, z_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(z_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale, bias = self.proj(z).chunk(2, dim=-1)  # each [B, C]
        return x * (1 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)


def _num_groups(channels: int, preferred: int = 8) -> int:
    g = min(preferred, channels)
    while channels % g != 0:
        g -= 1
    return g


class ResFiLMBlock1d(nn.Module):
    """2x upsample + residual conv block, conditioned on z at every block
    (not just at the input) via FiLM after each GroupNorm. GroupNorm instead
    of BatchNorm/InstanceNorm since num_groups doesn't depend on batch size
    (works fine at batch=1, unlike BatchNorm) and still normalizes per-sample
    (unlike BatchNorm's cross-sample statistics)."""

    def __init__(self, in_ch: int, out_ch: int, z_dim: int, kernel_size: int = 5, num_groups: int = 8):
        super().__init__()
        pad = kernel_size // 2
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(_num_groups(out_ch, num_groups), out_ch)
        self.film1 = FiLM1d(z_dim, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch, num_groups), out_ch)
        self.film2 = FiLM1d(z_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.act = nn.GELU()
        # Same residual-scale fix as ResUpBlock1d (see its comment) -- keeps
        # output variance from compounding across stacked blocks.
        self.res_scale = 1.0 / math.sqrt(2)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        skip = self.skip(x_up)
        h = self.film1(self.norm1(self.conv1(x_up)), z)
        h = self.act(h)
        h = self.film2(self.norm2(self.conv2(h)), z)
        return self.act((h + skip) * self.res_scale)


class WaveformUAPGeneratorResNet(nn.Module):
    """
    Conditional ResNet-style waveform generator: z_u is injected at every
    residual block (FiLM scale/bias), not only at the input projection --
    as opposed to WaveformUAPGeneratorConv, which only conditions once at
    the seed and lets the upsampling stack run unconditioned afterward.

    Input : z_u  [B, z_dim]
    Output: delta [B, patch_length]
    """

    def __init__(
        self,
        z_dim: int = 192,
        patch_length: int = 4096,
        base_channels: int = 1024,
        seed_len: int = 16,
        num_groups: int = 8,
    ):
        super().__init__()
        n_stages = int(math.log2(patch_length // seed_len))
        assert seed_len * (2 ** n_stages) == patch_length, (
            f"patch_length ({patch_length}) must be seed_len ({seed_len}) * a power of 2"
        )

        self.seed_len = seed_len
        self.base_channels = base_channels
        self.patch_length = patch_length

        self.proj = nn.Linear(z_dim, base_channels * seed_len)

        channels = [base_channels]
        for _ in range(n_stages):
            channels.append(max(32, channels[-1] // 2))

        self.blocks = nn.ModuleList([
            ResFiLMBlock1d(channels[i], channels[i + 1], z_dim, num_groups=num_groups)
            for i in range(n_stages)
        ])
        self.out_conv = nn.Conv1d(channels[-1], 1, kernel_size=7, padding=3)
        # Same small-output-init rationale as WaveformUAPGeneratorConv.
        nn.init.normal_(self.out_conv.weight, std=0.01)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
        Returns:
            delta: [B, patch_length]
        """
        B = z.shape[0]
        x = self.proj(z).view(B, self.base_channels, self.seed_len)
        for block in self.blocks:
            x = block(x, z)
        x = self.out_conv(x)  # [B, 1, patch_length]
        return x.squeeze(1)


class WaveformUAPGenerator(nn.Module):
    """
    Waveform-domain UAP generator: MLP-based direct output.

    Input : z_u  [B, z_dim]       speaker embedding (Zonos 128 + CAM++ 192 = 320)
    Output: delta [B, patch_length]  waveform-domain UAP patch

    Architecture: 3-layer MLP with GELU activations.
    """

    def __init__(
        self,
        z_dim: int = 320,
        hidden_dim: int = 2048,
        patch_length: int = 4096,
    ):
        super().__init__()
        self.patch_length = patch_length

        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.GELU(),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_length),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
        Returns:
            delta: [B, patch_length]
        """
        return self.net(z)
