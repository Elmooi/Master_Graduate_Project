# Third-Party Notices

This repository's own code is MIT-licensed (see `LICENSE`). It depends on the
following external models/libraries, which are **not vendored in this repo**
and must be installed/cloned separately:

| Dependency | License | Role | Source |
|---|---|---|---|
| Zonos | Apache-2.0 | One of four speaker encoders in the ensemble (`SpeakerEmbeddingLDA`) | `git clone https://github.com/Zyphra/Zonos.git && git checkout bc40d98` |
| CAM++ (via CosyVoice/ModelScope) | see upstream | One of four speaker encoders | https://github.com/modelscope/3D-Speaker |
| YourTTS speaker encoder (Coqui TTS) | MPL-2.0 | One of four speaker encoders | auto-downloaded by `coqui-tts` on first use |
| ECAPA-TDNN / ResNet ASV (SpeechBrain) | Apache-2.0 | Speaker-verification victim models used for evaluation | auto-downloaded via `speechbrain` |

The evaluation scripts in this repo also generate speech with several
third-party TTS systems as black-box "victim" targets (Tortoise-TTS, GPT-SoVITS,
Qwen3-TTS, XTTS-v2, StyleTTS2, Spark-TTS, Dia, etc.) and against the commercial
ElevenLabs API. None of their weights or code are included here — see each
system's own repository/terms for licensing.
