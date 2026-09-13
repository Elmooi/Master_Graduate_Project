# SUPER: Speaker-Conditioned Universal Perturbation Generator against Unauthorized Voice Cloning

SUPER protects a person's voice recordings from unauthorized zero-shot voice
cloning. A lightweight generator network is conditioned on a target speaker's
embedding and emits a short adversarial audio patch; tiling that patch across
a recording (at an inaudible SNR) is enough to make voice-cloning systems
produce speech that no longer matches the true speaker — **without any
per-recording optimization at protection time**. A single forward pass through
the generator (~0.2ms) replaces the iterative gradient-descent attack that
prior universal-perturbation methods require per speaker.

The generator is trained against an ensemble of four independently-trained
speaker-embedding models (**Zonos**, **CAM++**, **YourTTS's speaker encoder**,
**ECAPA-TDNN**) using **MLDG** (meta-learning for domain generalization) so
that the perturbation transfers to voice-cloning systems it never saw during
training. This repo evaluates black-box transfer against 9 independent TTS
systems plus the commercial ElevenLabs API.

This is a trimmed, portfolio-oriented copy of the full research repository:
it keeps the final model's training/protection/evaluation pipeline and
results, not every ablation and one-off experiment script from the project's
development history.

## Results

**Black-box protection rate** — fraction of protected recordings whose
voice-clone no longer matches the true speaker under an ECAPA-TDNN or ResNet
speaker verifier (higher is better; original, unprotected recordings score
near 0% by construction). Measured across 6 accent/dataset groups (VCTK,
LibriSpeech, Common Voice, plus three CSNED/CSUKIED/FST accent corpora),
70–300 held-out (never-trained-on) speaker/utterance pairs per system:

| Target TTS system | ECAPA protection | ResNet protection |
|---|---|---|
| GPT-SoVITS | 86.8% | 86.8% |
| Tortoise-TTS | 72.3% | 70.7% |
| Qwen3-TTS | 45.0% (59.3% @ th=0.45) | 41.3% (56.7% @ th=0.45) |
| XTTS-v2 (Coqui) | 60.7% | 53.9% |
| SV2TTS (Real-Time-Voice-Cloning) | 74.0% | 81.3% |
| StyleTTS2 | 68.9% | 73.5% |
| Spark-TTS | 91.8% | 93.2% |
| Dia-1.6B | 93.5% | 94.9% |
| ElevenLabs (commercial API) | 55.0% | 53.3% |

**Vs. Enkidu** (per-speaker optimization-based UAP baseline), same test
speakers/utterances, same verifiers:

| Target TTS system | Ours (ECAPA / ResNet) | Enkidu (ECAPA / ResNet) |
|---|---|---|
| XTTS-v2 | 74.9% / 68.9% | 54.8% / 39.3% |
| Spark-TTS | 91.8% / 93.2% | 45.1% / 40.8% |
| StyleTTS2 | 68.9% / 73.5% | 76.3% / 80.8% |
| Dia-1.6B | 93.5% / 94.9% | 83.8% / 80.6% |

Ours wins decisively against 3 of 4 systems and is competitive on the 4th
(StyleTTS2) — see `paper/SUPER.tex` for the full breakdown, ablations, and
discussion of the mixed StyleTTS2 result.

**Efficiency** (deployment-time cost, no gradient computation needed once the
generator is trained):

| Stage | Cost |
|---|---|
| Speaker-embedding extraction (3 reference clips, one-time per speaker) | 0.123 s |
| UAP generation (one forward pass) | 0.0002 s |
| Applying the perturbation to a 4s clip | 0.00025 s → real-time factor **0.000064** |

## Architecture

- `models/generator.py` — `WaveformUAPGenerator`: takes the concatenated
  four-encoder speaker embedding (1024-d) and emits a fixed-length waveform
  patch.
- `models/campplus_native.py` — a native PyTorch reimplementation of CAM++
  (one of the four speaker encoders), including a fix for a NaN-gradient bug
  in the original ONNX-traced version (a variance computed as `E[x²]-E[x]²`
  goes slightly negative under float cancellation on near-silent input,
  before the final `Sqrt` — clamped to `1e-12` here).
- `audio/tiler.py` — `WaveformTiler`: tiles the generated patch across an
  arbitrary-length recording at a target SNR (inaudibility constraint).
- `train_ensemble_full4_mldg.py` — trains the generator against the 4-encoder
  ensemble with an MLDG meta-learning objective for cross-system transfer.
- `protect_ensemble_full4.py` — applies a trained generator to a directory of
  speakers' recordings.
- `eval_zonos_only.py` — example black-box evaluation script: clones a
  protected/unprotected reference through a victim TTS system and measures
  the true-speaker cosine similarity via independent verifiers. The other 8
  TTS systems in the results table above use the same pattern against their
  own SDKs/APIs (not all included in this trimmed repo — see the paper for
  full methodology).

## Setup

Each TTS victim system needs its own conda environment due to conflicting
dependency pins; this repo ships `pip freeze` snapshots for the two
environments needed to run the included scripts:

```bash
conda create -n TTS python=3.9 && conda activate TTS
pip install -r environment/requirements-TTS.txt

conda create -n zonos python=3.10 && conda activate zonos
pip install -r environment/requirements-zonos.txt
```

You'll also need:
- [Zonos](https://github.com/Zyphra/Zonos) (Apache-2.0) — `git clone` and
  `checkout bc40d98`, then update `zonos_root` in `config.py` and the path in
  `eval_zonos_only.py`.
- A VoxCeleb1 copy for training (`voxceleb1_root` in `config.py`).

See `THIRD_PARTY_NOTICES.md` for the full dependency list.

## Usage

```bash
# 1. Train the generator against the 4-encoder MLDG ensemble
conda activate TTS
python train_ensemble_full4_mldg.py --epochs 500 --gpu 0 --speakers_per_batch 16

# 2. Apply the trained generator to a directory of speakers
python protect_ensemble_full4.py --ckpt checkpoints/ensemble_full4_mldg/generator_latest.pt

# 3. Evaluate black-box transfer against a victim TTS system
conda activate zonos
python eval_zonos_only.py --prot_root <protected_dir> --orig_root <original_dir>
```

## Paper

`paper/SUPER.tex` has the full methodology, all 9 TTS systems' results,
ablations, and the MLDG training details.

## License

MIT (this repository's own code) — see `LICENSE` and `THIRD_PARTY_NOTICES.md`
for dependencies.
