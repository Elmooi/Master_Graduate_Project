from __future__ import annotations  # 3.9 compat (TTS env) for `float | None` etc.

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    voxceleb1_root: str = "/path/to/voxceleb1"  # external dataset root -- update for your environment
    campplus_pt: str = "./weights/campplus/campplus_traced.pt"
    # GPU-traced variants (batch=1 only), one per physical GPU index —
    # traced constant tensors are baked to whatever device they were created
    # on and don't move with map_location/.to(device), so a single shared
    # file only works for the GPU it was traced on. train_campp.py picks
    # f"{campplus_pt_cuda_prefix}{local_rank}.pt".
    campplus_pt_cuda_prefix: str = "./weights/campplus/campplus_traced_cuda"
    campplus_fixed_t: int = 200   # trace 시 고정한 프레임 수
    # train_campp.py 전용: 200프레임(~2초)이 아니라 max_wav_seconds 전체를
    # 커버하도록 500프레임(~5초)으로 다시 트레이싱한 버전.
    campplus_pt_cuda_5s_prefix: str = "./weights/campplus/campplus_traced_5s_cuda"
    campplus_fixed_t_train: int = 500
    # train_ensemble.py only: CPU 5s-traced CAM++ (see train_ensemble.py's
    # campplus_embed docstring for why CAM++ runs on CPU there instead of the
    # GPU-traced *_cuda variants).
    campplus_pt_cpu_5s: str = "./weights/campplus/campplus_traced_5s_cpu.pt"
    # Batched variant (fixed batch=4, matching num_query_per_ep) -- CPU
    # single-utterance calls were the ensemble's bottleneck (~0.44s/call);
    # batching cuts per-call Python/TorchScript-interpreter overhead.
    # Numerically verified to match per-sample batch=1 calls to ~2.7e-6 (see
    # convert_campplus_cpu_5s_batch.py).
    campplus_pt_cpu_5s_batch4: str = "./weights/campplus/campplus_traced_5s_cpu_batch4.pt"
    # Same as above but with the E[x^2]-E[x]^2-then-Sqrt variance formula's
    # sole Sqrt op (xvector/stats/Sqrt) patched to clamp(min=1e-12) first --
    # fixes the root cause of CAM++'s repeated NaN-gradient bug (negative
    # variance from float cancellation on near-silent input). See
    # convert_campplus_patched.py.
    campplus_pt_cpu_5s_batch4_patched: str = "./weights/campplus/campplus_traced_5s_cpu_batch4_patched.pt"
    cache_dir: str = "./cache"
    ckpt_dir: str = "./checkpoints/campp_only"
    zonos_root: str = "/path/to/Zonos"  # git clone https://github.com/Zyphra/Zonos.git && git checkout bc40d98 (Apache-2.0)
    # RTVC (Resemblyzer-style GE2E) speaker encoder, borrowed from the
    # AudioLDM2-VoiceProtection project's EnsembleEncoders -- train_rtvc.py
    # only. Plain nn.Module (LSTM+Linear), no tracing/onnx needed unlike
    # CAM++.
    rtvc_root: str = "/path/to/rtvc"  # optional legacy encoder, unused by the ensemble_full4_mldg pipeline in this trimmed repo
    rtvc_encoder_ckpt: str = "/path/to/rtvc/encoder.pt"  # optional legacy encoder, unused by the ensemble_full4_mldg pipeline in this trimmed repo
    rtvc_z_dim: int = 256
    # YourTTS's speaker encoder (ResNet-SE, Coqui TTS) -- train_yourtts.py
    # only. Dropped RTVC: direct-delta ceiling test at the SAME 30dB budget
    # CAM++ uses reached cos 0.9722->0.4773 (near Zonos's ~0.47), vs RTVC's
    # 0.855 at 30dB (needed 20-25dB just to get partway there) -- YourTTS's
    # Conv+BatchNorm architecture has much healthier gradients than RTVC's
    # LSTM (no vanishing-gradient-through-time), so no special SNR/LR/
    # relation-loss handling needed, unlike RTVC.
    yourtts_se_config: str = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/config_se.json")  # auto-downloaded by coqui-tts on first use
    yourtts_se_ckpt: str = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_se.pth")  # auto-downloaded by coqui-tts on first use
    yourtts_z_dim: int = 512
    # relation-only grad norm (lambda=16) measured 1.36 vs adv-only 0.205 --
    # ~6.6x imbalance, same failure mode as RTVC (there it was 17x and
    # completely stalled training) just less severe. Disabling preemptively
    # rather than repeating that mistake.
    yourtts_lambda_relation: float = 0.0
    # adv-only grad norm (fp32-fixed) measured 0.205 at fresh init -- smaller
    # than CAM++'s typical 6-100+ that lr_train=6e-4 was tuned for, though
    # nowhere near RTVC's dead 0.012. val_adv plateaued at 0.744-0.746 for
    # epochs 19-49 while the direct-delta ceiling at this SNR is 0.477, i.e.
    # nowhere near the achievable floor yet -- unlike RTVC's genuine ceiling
    # hit, so trying a higher LR here is a reasonable next step rather than
    # assuming another ceiling.
    yourtts_lr_train: float = 1e-2

    # Ensemble (Zonos + CAM++ + ECAPA) -- train_ensemble.py only. ECAPA
    # (speechbrain/spkrec-ecapa-voxceleb) added as a 3rd target alongside the
    # original Zonos+CAM++ ensemble from train.py, matching Enkidu's
    # multi-encoder approach. Differentiability + fp16 autocast safety
    # confirmed directly (random-noise smoke test): no NaN, healthy grad
    # norm ~540 at fp16, unlike CAM++/YourTTS which both needed a forced-fp32
    # workaround for a clamp-underflow bug -- ECAPA doesn't appear to hit
    # that, but ecapa_embed() still forces fp32 defensively to match the
    # established pattern rather than trusting one synthetic smoke test.
    ecapa_z_dim: int = 192
    ensemble_lr_train: float = 6e-4
    ensemble_lambda_relation: float = 16.0

    # train_ensemble_yourtts.py: Zonos+CAM+++YourTTS (no ECAPA/speechbrain --
    # avoids the GPU corruption above entirely, since that was tied
    # specifically to co-loading speechbrain's ECAPA; uses native CAM++
    # (models/campplus_native.py) instead of the onnx2torch-traced model so
    # the whole thing runs in the `TTS` conda env, no onnx2torch needed).
    # lambda_relation reused as-is from the Zonos+CAM+++ECAPA run -- NOT
    # reverified for this new combo's gradient scale; if val_adv plateaus
    # unusually high early on, check for the same relation-loss-domination
    # pattern seen for RTVC/YourTTS/ECAPA (see ecapa_lambda_relation comment).
    ensemble2_lr_train: float = 6e-4
    ensemble2_lambda_relation: float = 16.0

    # train_ensemble_chatterbox.py: attacks the TWO speaker-conditioning
    # pathways Chatterbox's own TTS pipeline actually uses, discovered by
    # reading its source (chatterbox/tts.py's prepare_conditionals):
    #   - VoiceEncoder (ve.safetensors, GE2E-style 3-layer LSTM, adapted from
    #     CorentinJ/Real-Time-Voice-Cloning) -> feeds T3's speaker_emb, which
    #     drives the actual text-to-speech-TOKEN generation (the dominant
    #     driver of perceived voice identity)
    #   - CAM++ (s3gen.safetensors, same as campp_native_ckpt) -> feeds only
    #     s3gen's flow-matching vocoder conditioning
    # Earlier attempts only ever attacked the CAM++/s3gen half (even with
    # exact matching weights): protect rate against Chatterbox was ~12-17%
    # both with CosyVoice's CAM++ AND with Chatterbox's own CAM++ weights
    # (statistically indistinguishable), showing CAM++/s3gen's marginal
    # influence on final speaker identity is small regardless of weight
    # match -- VoiceEncoder/T3 (never attacked) evidently sets the ceiling.
    # This ensemble attacks both at once.
    voiceencoder_z_dim: int = 256
    chatterbox_target_snr_db: float | None = 20.0
    ensemble3_lr_train: float = 6e-4
    ensemble3_lambda_relation: float = 16.0
    # Measured (8 real speakers, isolated per-loss-term backward,
    # _diag_ensemble_chatterbox_balance.py): l_adv(campplus) alone produces
    # a generator gradient norm ~16.8x larger than l_adv(voiceencoder)
    # alone (mean 2.458 vs 0.1465) -- same recurring multi-target-scale-
    # mismatch pattern as RTVC (17x), YourTTS (6.6x), ECAPA relation-loss
    # (8.1x) elsewhere in this project. Without correction the shared
    # generator update would be dominated by CAM++ and VoiceEncoder would
    # barely train (matches the smoke test: val_campplus dropped to 0.20
    # after 1 epoch while val_voiceencoder stayed at 0.92). Downweight
    # campplus's adv loss term by ~1/16.8 to equalize contributions.
    ensemble3_campplus_weight: float = 0.06

    # train_ensemble_full4.py: Zonos + CAM++(onnx, CosyVoice's original
    # traced weights, NOT the native/Chatterbox one -- run on CPU/batched,
    # see campplus_embed_batch: co-loading it on GPU with speechbrain's
    # ECAPA reproduces the same TorchScript vector::_M_range_check
    # corruption found in train_ensemble.py's history) + YourTTS + ECAPA,
    # all 4 at once, no meta-learning (plain averaging, per user request).
    # Weights measured via _diag_ensemble_full4_balance.py (8 real
    # speakers, isolated per-loss-term grad norm, mean over speakers):
    # zonos=1.882, campplus=1.435, yourtts=0.5139, ecapa=0.4078 -- a much
    # milder spread (<=4.6x) than every other multi-target combo in this
    # project (RTVC 17x, YourTTS 6.6x, ECAPA relation 8.1x, ensemble3
    # 16.8x), but still corrected to equalize to zonos' scale.
    ensemble4_target_snr_db: float | None = 20.0
    ensemble4_lr_train: float = 6e-4
    # Measured (5 real batches, isolated grad norm, epoch38 checkpoint):
    # relation loss alone grad_norm~1.4 vs the (per-encoder-reweighted)
    # combined adv loss's BATCH-TOTAL grad_norm~27.7 (16 speakers summed,
    # since each speaker's adv backward adds to the same accumulated
    # gradient) -- relation was only ~5% (0.05x) of the adv total, i.e.
    # ~20x underweighted. This happened because ensemble4_{zonos,campplus,
    # yourtts,ecapa}_weight were calibrated to balance the 4 encoders
    # against EACH OTHER without re-checking relation's scale against the
    # new (larger) combined total -- same class of bug as every other
    # relation-loss mis-scaling this project hit, just inverted (under- not
    # over-weighted this time). Raised ~20x to restore parity. Continuing
    # training from the existing (epoch38) checkpoint rather than
    # restarting -- the adv-loss progress so far is legitimate, this only
    # corrects relation's influence going forward.
    ensemble4_lambda_relation: float = 320.0
    ensemble4_zonos_weight: float = 1.0
    ensemble4_campplus_weight: float = 1.311
    ensemble4_yourtts_weight: float = 3.662
    ensemble4_ecapa_weight: float = 4.614

    # train_ecapa.py only: ECAPA-alone target (ensemble abandoned -- CAM++'s
    # CPU model, needed to sidestep the GPU corruption from co-loading
    # speechbrain's ECAPA, kept SIGSEGVing under sustained multi-process CPU
    # load regardless of GPU/process count; see train_ensemble.py history).
    # ECAPA itself runs cleanly on GPU (fp32-forced, same as
    # yourtts_embed/ecapa_embed) with no analogous stability issues observed
    # in any test this session. lr/lambda_relation reused from CAM++'s
    # tuned values as a starting point (similar adv-grad-norm order of
    # magnitude measured: ECAPA ~0.30 vs CAM++ ~1.06 for one real episode).
    # 6e-4 -> 3e-3 didn't break the plateau (val_adv still ~0.75-0.78, same
    # SNR as CAM++ which reached ~0.42-0.53 -- so ECAPA is not inherently
    # unperturbable, just needs more push). Escalating further to 1e-2,
    # matching YourTTS's final escalation step that broke ITS plateau.
    ecapa_lr_train: float = 1e-2
    # Root cause of the 1000+ epoch plateau (val_adv stuck ~0.70-0.78 while
    # a direct-delta-optimization ceiling test hit cos~0.16 in just 300
    # steps on one speaker -- ECAPA is nowhere near unperturbable at this
    # SNR, the generator just wasn't learning to attack it). Measured
    # relation-loss-alone grad norm 2.18 vs adv-loss-alone 0.27 -- 8.1x
    # domination, same failure mode as RTVC (17x) and YourTTS (6.6x) from
    # reusing CAM++'s tuned lambda_relation=16 without checking. Disabled,
    # same fix as those two.
    ecapa_lambda_relation: float = 0.0
    # PCGrad projection removes conflicting gradient components, which changes
    # the effective update's magnitude/character vs plain averaging. Reusing
    # plain's lr=1e-2 caused val_adv to climb (worse) epoch9->29
    # (0.7181->0.7244->0.7328) with train_loss also rising -- instability,
    # not noise. Dropped 10x as a conservative restart point.
    ecapa_pcgrad_lr_train: float = 1e-3

    # Speaker subset
    num_speakers: int = 1000
    seed: int = 42

    # Support/query split (by video_id)
    num_support_videos: int = 3
    num_query_per_ep: int = 4

    # Audio
    sample_rate: int = 16000

    # Noise
    noise_level: float = 0.05
    target_snr_db: float | None = 30.0
    # train_rtvc.py only: direct-delta-optimization ceiling test showed RTVC
    # (LSTM-based, much more robust to additive noise than CAM++) barely
    # moves at 30dB (cos 0.998->0.855) and only partially at 25dB (->0.702);
    # 20dB was the strongest ceiling tested (->0.507, near Zonos's ~0.47)
    # but was ruled out as too audible. 25dB is the compromise.
    rtvc_target_snr_db: float | None = 25.0
    # train_yourtts.py only: direct-delta ceiling at 30dB reached cos 0.4773
    # (already fairly attackable), but per-user request, dropping to 20dB
    # (RTVC's strongest tested budget, ->0.507 there) for the MLDG experiment
    # to give the generator more room to work with.
    yourtts_target_snr_db: float | None = 20.0
    # train_ecapa.py only: matching the yourtts_target_snr_db drop (20dB
    # gave MLDG much more room to work with there), applied here too per
    # user request for the next ECAPA MLDG run.
    ecapa_target_snr_db: float | None = 20.0

    # Generator (WaveformUAPGenerator)
    z_dim: int = 128              # Zonos SpeakerEmbeddingLDA output dim
    campplus_z_dim: int = 192     # CAM++ output dim
    patch_length: int = 4096      # UAP waveform patch length (256 ms @ 16 kHz)
    gen_hidden_dim: int = 2048    # MLP hidden dim
    # train_campp.py only: doubled tile size vs patch_length (4096->8192,
    # halves the tiling repeat count across a wav). diag_gen_compare.py's
    # quick multi-speaker test (fp32 fixes, MLP, 16 speakers, 200 steps)
    # showed MLP@8192 converging at least as fast/clean as MLP@4096 (loss
    # 15.58->7.49 by step 20, 0 non-finite) -- no sign 8192 hurts, so trying
    # it in production instead of just 4096.
    patch_length_train: int = 8192

    @property
    def gen_z_dim(self) -> int:   # total generator input = Zonos + CAM++
        return self.z_dim + self.campplus_z_dim

    @property
    def ensemble_z_dim(self) -> int:   # Zonos + CAM++ + ECAPA
        return self.z_dim + self.campplus_z_dim + self.ecapa_z_dim

    @property
    def ensemble2_z_dim(self) -> int:   # Zonos + CAM++ + YourTTS
        return self.z_dim + self.campplus_z_dim + self.yourtts_z_dim

    @property
    def ensemble3_z_dim(self) -> int:   # Chatterbox's own VoiceEncoder + CAM++
        return self.voiceencoder_z_dim + self.campplus_z_dim

    @property
    def ensemble4_z_dim(self) -> int:   # Zonos + CAM++(onnx) + YourTTS + ECAPA
        return self.z_dim + self.campplus_z_dim + self.yourtts_z_dim + self.ecapa_z_dim

    # Training
    epochs: int = 500
    speakers_per_batch: int = 16
    lr: float = 1e-3
    # train_campp.py only, used with WaveformUAPGeneratorConv at
    # patch_length_train: the deeper stack (13 upsample stages vs 8 at the
    # default patch_length) showed grad norms in the hundreds of thousands to
    # millions at the default lr — lower this alongside the residual-scale
    # and small-output-init fixes in models/generator.py.
    # Once stable (residual scale + small init did their job — no more
    # crashes/explosions at 3e-4), val_adv was still only creeping down
    # ~0.001-0.002/epoch by epoch ~370 — decelerating hard, nowhere near a
    # useful level in reasonable time. Bumping back up partway to the
    # original 1e-3 to trade back some of that stability margin for speed
    # now that the pathological blowups are fixed at the source.
    lr_train: float = 6e-4
    # train_rtvc.py only: RTVC's 3-layer LSTM (vs CAM++'s TDNN/xvector, no
    # recurrence) has classic vanishing-gradient-through-time, worsened by
    # the ReLU right before the final L2-normalize. Measured grad norm on a
    # fresh generator (diag, 1 episode) was ~0.012 vs CAM++'s typical
    # 6-100+ at the same lr_train -- ~500-1000x smaller, so lr_train=6e-4
    # left RTVC essentially stuck (val_adv 0.9655->0.9639 over 30 epochs).
    rtvc_lr_train: float = 3e-2
    weight_decay: float = 1e-5
    lambda_perceptual: float = 0.0
    # Weight for the anti-collapse relation loss (train_campp.py only) —
    # matches pairwise delta-cosine structure to pairwise z_u-cosine
    # structure so the generator can't lazily converge to one
    # near-speaker-agnostic patch.
    # First try (1.0) reduced collapse (delta cos sim 0.73→0.61 by epoch 69/200)
    # but hadn't finished converging. l_adv gets backward() called once per
    # episode (speakers_per_batch=16 times per batch) while the relation loss
    # only gets called once per batch — with equal lambda, l_adv's gradient
    # contribution outnumbers it ~16:1. Scaling lambda_relation up to match
    # speakers_per_batch compensates for that imbalance.
    lambda_relation: float = 16.0
    # train_rtvc.py only: lambda_relation=16 was tuned for CAM++'s adv-loss
    # gradient scale. RTVC's adv-loss grad is ~500-1000x smaller (LSTM
    # vanishing gradient, see rtvc_lr_train comment) but the relation loss's
    # gradient doesn't touch the encoder at all, so it's the same magnitude
    # regardless of target -- measured grad norm from relation loss ALONE
    # (lambda=16) was 0.196 vs adv loss ALONE 0.012, i.e. relation loss
    # dominates by ~17x and the generator was mostly just satisfying that,
    # not actually fooling RTVC (val_adv stuck ~0.96-0.97 for 30+ epochs
    # regardless of lr_train). Disabled for RTVC.
    rtvc_lambda_relation: float = 0.0
    # train_campp.py only, training-time-only speaker-embedding noise
    # (train_campp.augment_z): tests whether the train/unseen downstream-
    # protection gap is caused by the generator memorizing the 999 exact
    # training z_u points instead of learning a locally-smooth mapping.
    # campplus_z_cache norms are typically ~0.7-0.85 (mean of several
    # normalized per-utterance embeddings, so not unit-norm) -- 0.03 is
    # roughly 4% of that scale, a small perturbation. 0.0 disables it.
    embed_aug_sigma: float = 0.03

    # train_campp.py --meta_learning only: MLDG (Li et al. 2018, "Learning
    # to Generalize: Meta-Learning for Domain Generalization") applied to
    # CAM++ training, treating each sampled speaker as one "domain". Not a
    # replacement for embed_aug_sigma above -- a different mechanism, tested
    # separately per explicit instruction.
    # Fraction of each sampled batch used as meta-train (S); the rest is
    # meta-test (T). 0.7 (e.g. 11/16 speakers) follows the roughly-2:1
    # source:held-out split MLDG's own experiments use.
    campp_meta_split_ratio: float = 0.7
    # --meta_learning uses a NATIVE (non-onnx2torch) CAM++ instead of the
    # traced GPU model run_epoch uses -- root cause, found via isolated
    # repro sweeping n_eps (number of CAM++ forward calls sharing one
    # create_graph=True graph): summing 3+ such calls produced NaN/huge
    # gradients in the SECOND-order backward probabilistically, reproduced
    # identically in BOTH the onnx2torch model (traced and untraced) AND a
    # from-scratch native reimplementation with real weights -- i.e. not an
    # onnx2torch bug, but CAM++'s statistics_pooling (var -> sqrt) having an
    # unbounded SECOND derivative as var->0 for specific channels/frames.
    # Fixed at the source in models/campplus_native.py (clamp(min=eps)
    # before sqrt, wiring up an eps param that used to be accepted but
    # unused) -- verified 0/60 failures across n_eps up to 11 after the fix,
    # vs. reliable failures before it. Weights: ResembleAI/chatterbox's
    # s3gen.safetensors speaker_encoder submodule (native PyTorch CAMPPlus
    # checkpoint, not CosyVoice's ONNX export -- see
    # models/campplus_native.py's statistics_pooling docstring and
    # campplus_native_chatterbox.pt).
    campp_native_ckpt: str = "./weights/campplus/campplus_native_chatterbox.pt"
    # Inner-loop virtual-SGD step size (alpha) used to build the fast
    # weights theta' = theta - alpha * grad(L_train). Independent of the
    # real optimizer's lr_train -- this one only exists inside the
    # (never-applied) virtual update, so it can be tuned separately.
    campp_meta_inner_lr: float = 1e-2
    # Weight (beta) on the meta-test term in the final objective
    # L_train(theta) + beta * L_test(theta'). 1.0 matches the original
    # MLDG paper's equal weighting.
    campp_meta_beta: float = 1.0

    # train_yourtts.py --meta_learning: same MLDG hyperparameters as CAM++'s,
    # reused as-is (no separate tuning done for YourTTS).
    yourtts_meta_split_ratio: float = 0.7
    yourtts_meta_inner_lr: float = 1e-2
    yourtts_meta_beta: float = 1.0
    # train_ensemble_full4_mldg.py: same MLDG hyperparameters reused as-is,
    # UNVERIFIED for the 4-encoder ensemble combo -- inner_lr and
    # lambda_relation were calibrated for the non-meta ensemble's gradient
    # scale, which changes once an inner virtual-update step is added.
    ensemble4_meta_split_ratio: float = 0.7
    ensemble4_meta_inner_lr: float = 1e-2
    ensemble4_meta_beta: float = 1.0
    # YourTTS's ResNetSpeakerEncoder ASP pooling computes
    # sqrt((E[x^2]-E[x]^2).clamp(min=1e-5)) -- the exact same
    # variance->sqrt pattern as CAM++'s statistics_pooling, with an eps
    # (1e-5) far too small to bound the SECOND derivative under
    # --meta_learning's create_graph=True double-backward (see
    # campp_native_ckpt's comment above for the full root-cause story).
    # Since this is third-party code (TTS.encoder.models.resnet), fixed via
    # a runtime monkeypatch of the encoder instance's forward() (only when
    # --meta_learning is set; the plain run_epoch path uses the untouched
    # original) rather than editing site-packages. Same eps value as
    # campplus_native.py's fix.
    yourtts_asp_eps: float = 1e-2

    # train_ecapa.py --meta_learning: same MLDG hyperparameters, reused as-is.
    ecapa_meta_split_ratio: float = 0.7
    ecapa_meta_inner_lr: float = 1e-2
    ecapa_meta_beta: float = 1.0
    # speechbrain's ECAPA_TDNN AttentiveStatisticsPooling has the exact same
    # variance->sqrt(clamp(eps)) pattern (eps=1e-12, even smaller than
    # YourTTS's 1e-5) -- same double-backward instability risk under
    # --meta_learning. Unlike YourTTS's inline ASP code, speechbrain stores
    # eps as a plain instance attribute (self.eps, read fresh each forward),
    # so the fix here is just mutating that attribute post-construction
    # (no forward() rewrite needed) -- see train_ecapa.py's
    # _patch_ecapa_asp_eps.
    #
    # NOT 1e-2 (CAM++/YourTTS's value) -- reusing that blindly here caused a
    # real (silent, non-crashing) failure: measured intra/inter-speaker
    # cosine separation on 6 real speakers collapsed from 0.79 (eps<=1e-4,
    # unchanged from the original 1e-12) to 0.51 at eps=1e-3, 0.06 at 3e-3,
    # and 0.01 at 1e-2 -- i.e. eps=1e-2 makes ECAPA's own embeddings nearly
    # speaker-independent, so val_adv stuck at ~0.97-1.0 for 50 epochs
    # wasn't a training bug, the "attack target" itself had been destroyed.
    # 1e-4 preserves separation essentially exactly (0.788 vs 0.7884
    # baseline) -- ECAPA's real per-channel variance during pooling is
    # evidently much smaller than CAM++'s/YourTTS's, so the same eps that
    # was "small" for them is not small for this encoder.
    ecapa_asp_eps: float = 1e-4

    # Train / val split
    val_ratio: float = 0.2

    # Memory
    max_wav_seconds: float = 5.0

    # Eval
    num_eval_speakers: int = 50
    eval_every: int = 10


CFG = Config()
