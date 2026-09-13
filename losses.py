"""
Loss functions for Speaker-conditioned UAP Generator.

l_adv: pairwise clean-vs-protected cosine similarity (per utterance).
l_per: normalized MSE in the waveform domain (Enkidu-identical).
"""

import torch
import torch.nn.functional as F


def l_adv(emb_protected: torch.Tensor, emb_clean: torch.Tensor) -> torch.Tensor:
    """
    Adversarial loss: push protected embedding away from its own clean embedding.

    Minimise cosine similarity between E(x̃) and E(x).detach() so that
    gradients flow only through the protected path, not the clean path.

    Args:
        emb_protected: [batch, D]  embedding of perturbed audio E(x̃)
        emb_clean:     [batch, D]  embedding of clean audio E(x), detached

    Returns:
        scalar loss (mean over batch)
    """
    return F.cosine_similarity(
        emb_protected,
        emb_clean.detach().to(emb_protected.dtype),
        dim=-1,
    ).mean()


def l_per(
    clean_waveform: torch.Tensor,
    noisy_waveform: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Perceptual loss — exact Enkidu implementation (core/enkidu.py:244).
    Normalised MSE (SNR inverse) in the waveform domain.

    Args:
        clean_waveform: [B, N]
        noisy_waveform: [B, N]

    Returns:
        scalar loss
    """
    diff = noisy_waveform - clean_waveform
    num = (diff ** 2).mean(dim=-1)
    den = (clean_waveform ** 2).mean(dim=-1) + eps
    return (num / den).mean()


def total_loss(
    emb_protected: torch.Tensor,
    emb_clean: torch.Tensor,
    clean_waveform: torch.Tensor,
    noisy_waveform: torch.Tensor,
    lambda_perceptual: float = 0.1,
) -> torch.Tensor:
    adv = l_adv(emb_protected, emb_clean)
    per = l_per(clean_waveform, noisy_waveform)
    return adv + lambda_perceptual * per
