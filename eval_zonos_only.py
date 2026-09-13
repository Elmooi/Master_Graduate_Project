#!/usr/bin/env python3
"""
Zonos TTS로 protected/original 참조음성 voice cloning 평가.

{prot_root}/{dataset}_{speaker}/ 화자들에 대해:
  1. 원본(unprotected) 참조음성으로 Zonos TTS 생성
  2. 보호(protected) 참조음성으로 Zonos TTS 생성 (같은 대사)
  3. 두 TTS 결과와 "진짜 화자"(원본 wav 평균 임베딩) 간 ECAPA/ResNet cosine 유사도 비교
     -> 보호TTS 유사도가 원본TTS보다 많이 낮을수록 보호가 잘 되는 것.

eval_campp_only.py(CosyVoice 버전)와 동일 구조, TTS 엔진만 Zonos로 교체.
speaker conditioning은 Zonos의 make_speaker_embedding()으로 참조 wav에서 직접 추출
(icassp2027 학습에 쓰인 SpeakerEmbeddingLDA z_u caching과는 별개, TTS 엔진 자체의
voice-cloning 조건화 경로).

Must run in the `zonos` conda env.

Usage:
    python  (in the "zonos" conda env) eval_zonos_only.py \
        --prot_root protected_test_output_noise011 \
        --out_root eval_tts_output_noise011_zonos [--n_per_spk 3]
"""
import argparse
import csv
import random
import sys
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=0)
_gpu_args, _ = _pre.parse_known_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_args.gpu)

sys.path.insert(0, "/path/to/Zonos")  # git clone https://github.com/Zyphra/Zonos.git && git checkout bc40d98 (Apache-2.0)
os.chdir("/path/to/Zonos")

import torch
import torch.nn.functional as F
import torchaudio

PROT_ROOT = Path("./protected_test_output_campp")
ORIG_ROOT = Path("./unprotected_test_dataset")
SCRIPT_PATH = Path("./script.txt")  # any newline-separated list of held-out English sentences
OUT_ROOT = Path("./eval_tts_output_zonos")
ECAPA_CACHE = os.path.expanduser("~/.cache/speechbrain/ecapa")
RESNET_CACHE = os.path.expanduser("~/.cache/speechbrain/resnet")
ZONOS_MODEL = "Zyphra/Zonos-v0.1-transformer"
TARGET_SR = 16000
SEED = 42


def load_wav(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_per_spk", type=int, default=3, help="화자당 평가할 발화 수")
    parser.add_argument("--prot_root", default=str(PROT_ROOT))
    parser.add_argument("--orig_root", default=str(ORIG_ROOT))
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    args = parser.parse_args()
    prot_root = Path(args.prot_root)
    orig_root = Path(args.orig_root)
    out_root = Path(args.out_root)

    device = torch.device("cuda")

    print("Loading ECAPA-TDNN...")
    from speechbrain.pretrained import EncoderClassifier
    ecapa_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ECAPA_CACHE,
        run_opts={"device": str(device)},
    )
    print("Loading ResNet (spkrec-resnet-voxceleb)...")
    resnet_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-resnet-voxceleb",
        savedir=RESNET_CACHE,
        run_opts={"device": str(device)},
    )

    def make_embed_fn(encoder):
        def embed_fn(wav: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
            if sr != TARGET_SR:
                wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            with torch.no_grad():
                emb = encoder.encode_batch(wav.to(device))
            return F.normalize(emb.squeeze(0).squeeze(0), dim=-1)
        return embed_fn

    verifiers = {"ecapa": make_embed_fn(ecapa_encoder), "resnet": make_embed_fn(resnet_encoder)}

    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    print("Loading Zonos...")
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    zonos_model = Zonos.from_pretrained(ZONOS_MODEL, device=device)
    zonos_sr = zonos_model.autoencoder.sampling_rate

    def tts(text: str, ref_wav_path: Path) -> torch.Tensor:
        wav_ref, sr_ref = torchaudio.load(str(ref_wav_path))
        speaker = zonos_model.make_speaker_embedding(wav_ref, sr_ref)
        cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
        conditioning = zonos_model.prepare_conditioning(cond_dict)
        codes = zonos_model.generate(conditioning)
        wav_out = zonos_model.autoencoder.decode(codes).cpu()[0]  # [1, T]
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)
        return wav_out

    with open(SCRIPT_PATH, encoding="utf-8") as f:
        scripts = [l.strip() for l in f if l.strip()]

    rng = random.Random(SEED)
    spk_dirs = sorted(prot_root.glob("*/"))
    print(f"\n{len(spk_dirs)} protected speaker folders found.\n")

    out_root.mkdir(parents=True, exist_ok=True)
    results = []

    known_datasets = sorted(
        (d.name for d in orig_root.iterdir() if d.is_dir()),
        key=len, reverse=True,
    )

    for spk_dir in spk_dirs:
        folder_name = spk_dir.name  # "{dataset}_{speaker}"
        dataset = next((d for d in known_datasets if folder_name.startswith(d + "_")), None)
        if dataset is None:
            print(f"[SKIP] unrecognized dataset prefix: {folder_name}")
            continue
        speaker = folder_name[len(dataset) + 1:]
        orig_dir = orig_root / dataset / speaker
        if not orig_dir.is_dir():
            print(f"[SKIP] no original dir for {folder_name}: {orig_dir}")
            continue

        prot_wavs = sorted(spk_dir.glob("*.wav"))
        if not prot_wavs:
            continue
        chosen = rng.sample(prot_wavs, min(args.n_per_spk, len(prot_wavs)))

        orig_all_wavs = sorted(orig_dir.glob("*.wav"))
        true_embs = {
            name: torch.stack([fn(load_wav(p)) for p in orig_all_wavs]).mean(0)
            for name, fn in verifiers.items()
        }

        print(f"\n{'='*60}\n[{folder_name}]  ({len(chosen)}개 평가)")

        for prot_path in chosen:
            fname = prot_path.name
            orig_path = orig_dir / fname
            if not orig_path.exists():
                print(f"  [SKIP] no matching original: {fname}")
                continue

            text = rng.choice(scripts)
            out_dir = out_root / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)
            tts_orig_path = out_dir / f"{prot_path.stem}_tts_orig.wav"
            tts_prot_path = out_dir / f"{prot_path.stem}_tts_prot.wav"

            try:
                torch.manual_seed(SEED)
                wav_tts_orig = tts(text, orig_path)
                torch.manual_seed(SEED)
                wav_tts_prot = tts(text, prot_path)
                torchaudio.save(str(tts_orig_path), wav_tts_orig, zonos_sr)
                torchaudio.save(str(tts_prot_path), wav_tts_prot, zonos_sr)

                print(f"  [{fname}] text='{text[:40]}...'")
                row = dict(dataset=dataset, speaker=speaker, file=fname, text=text[:50])
                for name, fn in verifiers.items():
                    emb_tts_orig = fn(wav_tts_orig, sr=zonos_sr)
                    emb_tts_prot = fn(wav_tts_prot, sr=zonos_sr)
                    s_orig = cos_sim(true_embs[name], emb_tts_orig)
                    s_prot = cos_sim(true_embs[name], emb_tts_prot)
                    print(f"    [{name}] 원본-ref: {s_orig:.4f}  보호-ref: {s_prot:.4f}  (Δ={s_prot - s_orig:+.4f})")
                    row[f"score_orig_{name}"] = round(s_orig, 4)
                    row[f"score_prot_{name}"] = round(s_prot, 4)
                row["score_orig"] = row["score_orig_ecapa"]
                row["score_prot"] = row["score_prot_ecapa"]

                results.append(row)
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")

    print(f"\n{'='*60}\n【 전체 결과 요약 】")
    if results:
        n = len(results)
        print(f"  평가 파일 수: {n}")
        for name in verifiers:
            orig_avg = sum(r[f"score_orig_{name}"] for r in results) / n
            prot_avg = sum(r[f"score_prot_{name}"] for r in results) / n
            print(f"  [{name}] 원본-ref 평균: {orig_avg:.4f}  보호-ref 평균: {prot_avg:.4f}  "
                  f"보호 효과: {orig_avg - prot_avg:+.4f}")

        out_csv = out_root / "eval_result.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"  결과 저장: {out_csv}")
    else:
        print("  평가된 파일이 없습니다.")


if __name__ == "__main__":
    main()
