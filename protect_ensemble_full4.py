"""
4-way ensemble (Zonos+CAM+++YourTTS+ECAPA) 체크포인트로
unprotected_test_dataset의 랜덤 화자들에 UAP 노이즈를 입혀 저장.

protect_campp_only.py/protect_ecapa_only.py와 동일 구조, z_u만 4개 인코더
concat(1024차원)으로 교체.

Output: protected_test_output_ensemble_full4/{dataset}_{speaker_id}/{file}.wav

Must run in the `TTS` conda env (has zonos, onnx2torch, speechbrain, coqui-tts
all installed -- see train_ensemble_full4.py's docstring).

Usage:
    conda run -n TTS python protect_ensemble_full4.py [--n_speakers 42] [--device cuda:0]
"""
import argparse
import random
from pathlib import Path

import torch
import torchaudio

from config import CFG
from audio.tiler import WaveformTiler
from models.generator import WaveformUAPGenerator
from train_ensemble_full4 import (
    Z_DIM, zonos_embed, campplus_embed_batch, ecapa_embed,
    SpeakerEmbeddingLDA,
)
from train_yourtts import build_yourtts_encoder, yourtts_embed

TEST_ROOT = Path("./unprotected_test_dataset")
OUT_ROOT  = Path("./protected_test_output_ensemble_full4")
CKPT_PATH = Path("./checkpoints/ensemble_full4/generator_latest.pt")
SEED       = 42
N_REF_WAVS = 3


def load_wav(path: Path, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def collect_speakers(root: Path) -> list[tuple[str, str, Path]]:
    entries = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for spk_dir in sorted(dataset_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            if list(spk_dir.glob("*.wav")):
                entries.append((dataset_dir.name, spk_dir.name, spk_dir))
    return entries


@torch.no_grad()
def compute_speaker_z(zonos_enc, campplus, yourtts_enc, ecapa, spk_dir: Path, device: torch.device, n_ref: int = N_REF_WAVS) -> torch.Tensor:
    wavs = sorted(spk_dir.glob("*.wav"))[:n_ref]
    zonos_embs, yourtts_embs, ecapa_embs = [], [], []
    raw_wavs = []
    for wp in wavs:
        try:
            wav = load_wav(wp, CFG.sample_rate).to(device)
            raw_wavs.append(wav)
            zonos_embs.append(zonos_embed(wav, zonos_enc).squeeze(0))
            yourtts_embs.append(yourtts_embed(wav.float(), yourtts_enc, device).squeeze(0))
            ecapa_embs.append(ecapa_embed(wav, ecapa, device).squeeze(0))
        except Exception as e:
            print(f"  [warn] skipping {wp.name}: {e}")
    if not raw_wavs:
        raise RuntimeError(f"No valid reference wavs in {spk_dir}")

    campplus_embs = [campplus_embed_batch(w, campplus, device).squeeze(0) for w in raw_wavs]

    z_zonos = torch.stack(zonos_embs).mean(0)
    z_campplus = torch.stack(campplus_embs).mean(0)
    z_yourtts = torch.stack(yourtts_embs).mean(0)
    z_ecapa = torch.stack(ecapa_embs).mean(0)
    return torch.cat([z_zonos, z_campplus, z_yourtts, z_ecapa], dim=0)  # [1024]


def protect_speaker(generator, tiler, z_u, spk_dir: Path, out_dir: Path, device: torch.device) -> None:
    with torch.no_grad():
        delta = generator(z_u.unsqueeze(0))  # [1, patch_length]

    out_dir.mkdir(parents=True, exist_ok=True)
    for wp in sorted(spk_dir.glob("*.wav")):
        try:
            wav = load_wav(wp, CFG.sample_rate).to(device)
            with torch.no_grad():
                protected = tiler(wav, delta)
            torchaudio.save(str(out_dir / wp.name), protected.cpu(), CFG.sample_rate)
            print(f"    saved {wp.name}")
        except Exception as e:
            print(f"    [warn] failed {wp.name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_speakers", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--ckpt", default=str(CKPT_PATH))
    parser.add_argument("--test_root", default=str(TEST_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)
    ckpt_path = Path(args.ckpt)
    test_root = Path(args.test_root)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading Zonos encoder...")
    zonos_enc = SpeakerEmbeddingLDA(device=str(device)).to(device).eval()

    print("Loading CAM++ encoder (onnx2torch-traced, CPU)...")
    campplus = torch.jit.load(CFG.campplus_pt_cpu_5s_batch4_patched, map_location="cpu").eval()
    for p in campplus.parameters():
        p.requires_grad_(False)

    print("Loading YourTTS encoder...")
    yourtts_enc = build_yourtts_encoder(device)

    print("Loading ECAPA encoder...")
    from speechbrain.pretrained import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain/ecapa"),
        run_opts={"device": str(device)},
    )

    print("Loading generator checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator = WaveformUAPGenerator(
        z_dim=Z_DIM,
        hidden_dim=CFG.gen_hidden_dim,
        patch_length=CFG.patch_length_train,
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print(f"  resumed from epoch {ckpt['epoch']}")

    tiler = WaveformTiler(noise_level=CFG.noise_level, target_snr_db=CFG.ensemble4_target_snr_db)

    all_speakers = collect_speakers(test_root)
    print(f"Total speakers found: {len(all_speakers)}")
    random.seed(SEED)
    selected = random.sample(all_speakers, min(args.n_speakers, len(all_speakers)))

    print(f"\nSelected {len(selected)} speakers:")
    for ds, spk, _ in selected:
        print(f"  {ds}/{spk}")

    out_root.mkdir(parents=True, exist_ok=True)

    for ds, spk, spk_dir in selected:
        out_dir = out_root / f"{ds}_{spk}"
        print(f"\n[{ds}/{spk}]")
        try:
            z_u = compute_speaker_z(zonos_enc, campplus, yourtts_enc, ecapa, spk_dir, device)
            print(f"  z_u norm: {z_u.norm():.4f}")
            protect_speaker(generator, tiler, z_u, spk_dir, out_dir, device)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\nDone. Output saved to: {out_root}")


if __name__ == "__main__":
    main()
