"""
VoxCeleb1 dataset for speaker-conditioned UAP training.

Folder structure (VoxCeleb1):
    {root}/{speaker_id}/{video_id}/{utterance_id}.wav

Support / query split:
    The first `num_support_videos` video_ids (sorted) → support set.
    Remaining video_ids → query set.

Train / val split within query:
    Query wavs are sorted (reproducible) then split by val_ratio.
    First (1 - val_ratio) fraction → train_query.
    Last val_ratio fraction         → val_query.
    __getitem__ samples only from train_query.
"""

from __future__ import annotations  # 3.9 compat (TTS env)

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset


def _collect_speaker_videos(
    root: Path,
    speaker_ids: List[str],
) -> Dict[str, Dict[str, List[Path]]]:
    data: Dict[str, Dict[str, List[Path]]] = {}
    for spk in speaker_ids:
        spk_dir = root / spk
        if not spk_dir.is_dir():
            continue
        videos: Dict[str, List[Path]] = {}
        for vid_dir in sorted(spk_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            wavs = sorted(vid_dir.glob("*.wav"))
            if wavs:
                videos[vid_dir.name] = wavs
        if len(videos) >= 2:
            data[spk] = videos
    return data


class VoxCeleb1EpisodeDataset(Dataset):
    """
    Each item = one training episode for one speaker.
    __getitem__ samples from train_query only.
    val_query is exposed for evaluation.
    """

    def __init__(
        self,
        root: str,
        num_speakers: int = 100,
        num_support_videos: int = 3,
        num_query_per_ep: int = 4,
        seed: int = 42,
        exclude_speakers: set[str] | None = None,
        val_ratio: float = 0.2,
    ):
        super().__init__()
        self.root = Path(root)
        self.num_support_videos = num_support_videos
        self.num_query_per_ep = num_query_per_ep

        rng = random.Random(seed)
        all_speakers = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if exclude_speakers:
            all_speakers = [s for s in all_speakers if s not in exclude_speakers]
        selected = rng.sample(all_speakers, min(num_speakers, len(all_speakers)))

        raw = _collect_speaker_videos(self.root, selected)

        self.speakers:     List[str]              = []
        self.support:      Dict[str, List[Path]]  = {}
        self.train_query:  Dict[str, List[Path]]  = {}
        self.val_query:    Dict[str, List[Path]]  = {}

        for spk, videos in raw.items():
            vid_names = sorted(videos.keys())
            if len(vid_names) <= num_support_videos:
                support_vids = vid_names[:len(vid_names) - 1]
                query_vids   = vid_names[len(vid_names) - 1:]
            else:
                support_vids = vid_names[:num_support_videos]
                query_vids   = vid_names[num_support_videos:]

            support_wavs: List[Path] = []
            for v in support_vids:
                support_wavs.extend(videos[v])

            query_wavs: List[Path] = []
            for v in query_vids:
                query_wavs.extend(videos[v])

            if not support_wavs or not query_wavs:
                continue

            # Fixed train/val split — sorted for reproducibility, no extra seed needed
            sorted_query = sorted(query_wavs, key=str)
            n_val = max(1, int(len(sorted_query) * val_ratio))
            n_train = len(sorted_query) - n_val
            if n_train < 1:
                continue

            self.speakers.append(spk)
            self.support[spk]     = support_wavs
            self.train_query[spk] = sorted_query[:n_train]
            self.val_query[spk]   = sorted_query[n_train:]

        self._rng = random.Random(seed + 1)

    def __len__(self) -> int:
        return len(self.speakers)

    def __getitem__(self, idx: int) -> dict:
        spk = self.speakers[idx]
        pool = self.train_query[spk]
        n = min(self.num_query_per_ep, len(pool))
        sampled = self._rng.sample(pool, n)
        return {
            "speaker_id":    spk,
            "support_paths": self.support[spk],
            "query_paths":   sampled,
        }


def load_wav(path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load a wav file and resample to target_sr. Returns [1, N]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def episode_collate(batch):
    return batch
