"""
Ensemble (Zonos + CAM++[onnx] + YourTTS + ECAPA) Speaker-conditioned UAP
Generator training WITH MLDG meta-learning.

Direct combination of train_ensemble_full4.py (4-encoder weighted embedding
loss + relation loss, CPU CAM++) with train_yourtts.py's run_epoch_mldg
(speaker-level meta-train/meta-test split, single virtual gradient step,
full second-order differentiation through the inner update). This exact
combination was never trained before this script -- see the symbolic
derivation worked out in conversation (2.3.2-equivalent) for the objective.

z_u = concat(Zonos[128], CAM++[192], YourTTS[512], ECAPA[192]) = [1024]

Per-episode loss (meta-train and meta-test both use this, meta-test with
fast_weights substituted in):
  ell(phi; z_u, wavs) = sum_k w_k * l_adv(E_k(protected), E_k(clean))
  [+ lambda_perceptual * l_per, if > 0 -- currently 0]

Meta objective per batch (P speakers, split ratio rho into train/test):
  L_train(theta)  = mean over meta-train speakers of ell(theta; ...)
  theta'          = theta - eta * grad_theta L_train(theta)   [1 step, create_graph=True]
  L_test(theta')  = mean over meta-test speakers of ell(theta'; ...)
  L_rel(theta)    = relation loss over the FULL batch (train+test), using
                    theta (not theta'), computed once, excluded from the
                    inner update -- identical semantics to
                    train_ensemble_full4.py's non-meta relation loss.
  L_meta(theta)   = L_train(theta) + beta*L_test(theta') + lambda_rel*L_rel(theta)

No AMP/GradScaler here (matches train_yourtts.py's run_epoch_mldg): the
double-backward through create_graph=True interacts badly with fp16
autocast, so this runs in full fp32 throughout -- slower per batch than
train_ensemble_full4.py's plain run_epoch, as expected for MLDG.

Single-GPU only (no DDP) -- MLDG's manual dist.all_reduce path was only
ever exercised for the single-encoder trainers and adds complexity not
worth it for an ablation run.

Usage (single GPU, warm-started from the existing non-MLDG ensemble
checkpoint's generator weights only -- optimizer/epoch NOT restored,
this begins a fresh MLDG training run from a good initialization):
    python  (in the "TTS" conda env) train_ensemble_full4_mldg.py \
        --epochs 500 --init_from checkpoints/ensemble_full4/generator_latest.pt

Usage (resuming an in-progress MLDG run):
    python  (in the "TTS" conda env) train_ensemble_full4_mldg.py \
        --epochs 500 --resume checkpoints/ensemble_full4_mldg/generator_latest.pt
"""

import argparse
import faulthandler
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

from config import CFG

sys.path.insert(0, CFG.zonos_root)
from zonos.speaker_cloning import SpeakerEmbeddingLDA

from audio.tiler import WaveformTiler
from data.voxceleb import VoxCeleb1EpisodeDataset, load_wav
from losses import l_adv, l_per
from models.generator import WaveformUAPGenerator
from train_yourtts import build_yourtts_encoder, yourtts_embed
from train_ensemble_full4 import (
    Z_DIM, load_z_cache, augment_z, zonos_embed, campplus_embed_batch,
    ecapa_embed, embed_all, run_val,
)


def delta_and_upstream_grad(z_u, raw_wavs, generator, weights, zonos_enc, campplus,
                             yourtts_enc, ecapa, tiler, device, max_wav_len, fast_weights=None):
    """
    Returns (loss_value: float, delta: Tensor[still connected to generator/fast_weights
    graph], upstream_grad: Tensor[detached] = d(loss)/d(delta)).

    Memory-efficiency trick: the frozen speaker encoders (large: ECAPA, YourTTS, Zonos,
    CAM++) are run on a DETACHED copy of delta (a fresh leaf), then a normal (non-create_graph)
    backward computes d(loss)/d(delta) and immediately frees all of the encoders' internal
    activations, exactly like the plain non-meta trainer does per-speaker. Only `delta`
    itself is left connected to the (small, ~18M-param MLP) generator's own forward graph.

    This means a later create_graph=True differentiation of `delta` w.r.t. generator params
    (see run_epoch_mldg) never has to build double-backward-capable buffers through the
    encoders at all -- confirmed by direct experiment (see conversation) that naively
    computing the whole episode loss under create_graph=True OOMs a 24GB GPU even at the
    smallest possible batch (1 meta-train speaker, 2 query utterances, 2s clips), because
    PyTorch has no way to know the encoders don't need a second derivative and keeps their
    full activations alive anyway.
    """
    min_len = min(w.shape[0] for w in raw_wavs)
    wavs = torch.stack([w[:min_len] for w in raw_wavs])  # [Q, N]
    Q = wavs.shape[0]
    z_batch = z_u.unsqueeze(0).expand(Q, -1)

    if fast_weights is None:
        delta = generator(z_batch)
    else:
        delta = torch.func.functional_call(generator, fast_weights, (z_batch,))

    delta_leaf = delta.detach().requires_grad_(True)
    protected = tiler(wavs, delta_leaf)

    with torch.no_grad():
        emb_zc, emb_cc, emb_yc, emb_ec = embed_all(wavs, zonos_enc, campplus, yourtts_enc, ecapa, device)
    emb_zp, emb_cp, emb_yp, emb_ep = embed_all(protected, zonos_enc, campplus, yourtts_enc, ecapa, device)

    loss = (
        weights[0] * l_adv(emb_zp, emb_zc)
        + weights[1] * l_adv(emb_cp, emb_cc)
        + weights[2] * l_adv(emb_yp, emb_yc)
        + weights[3] * l_adv(emb_ep, emb_ec)
    )
    if CFG.lambda_perceptual > 0.0:
        loss = loss + CFG.lambda_perceptual * l_per(wavs, protected)

    loss.backward()  # cheap, normal backward -- only touches delta_leaf and upstream (encoders)
    upstream_grad = delta_leaf.grad.detach().clone()
    return loss.item(), delta, upstream_grad


def run_epoch_mldg(
    generator, zonos_enc, campplus, yourtts_enc, ecapa,
    tiler: WaveformTiler,
    dataset: VoxCeleb1EpisodeDataset,
    zonos_z_cache: dict, campplus_z_cache: dict, yourtts_z_cache: dict, ecapa_z_cache: dict,
    optimizer: optim.Optimizer,
    device: torch.device,
    speakers_per_batch: int,
    lambda_relation: float,
    meta_split_ratio: float,
    meta_inner_lr: float,
    meta_beta: float,
    max_wav_len: int,
    epoch: int,
    ckpt_dir: Path,
) -> float:
    generator.train()

    rng = random.Random(epoch * 31337)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    caches = (zonos_z_cache, campplus_z_cache, yourtts_z_cache, ecapa_z_cache)
    weights = (CFG.ensemble4_zonos_weight, CFG.ensemble4_campplus_weight,
               CFG.ensemble4_yourtts_weight, CFG.ensemble4_ecapa_weight)

    params = list(generator.parameters())
    param_names = [name for name, _ in generator.named_parameters()]

    total_loss_val = 0.0
    num_batches = 0

    batch_ranges = range(0, len(indices), speakers_per_batch)
    batch_iter = tqdm(batch_ranges, desc=f"Ep {epoch:03d} [MLDG]", leave=False, unit="batch")
    for batch_start in batch_iter:
        batch_indices = indices[batch_start:batch_start + speakers_per_batch]
        episodes = [dataset[i] for i in batch_indices]
        if not episodes:
            continue

        valid_eps = []
        for ep in episodes:
            spk = ep["speaker_id"]
            if not all(spk in c for c in caches):
                continue
            z_u = torch.cat([c[spk].to(device) for c in caches], dim=0)  # [1024]
            z_u = augment_z(z_u, CFG.embed_aug_sigma)

            raw_wavs = []
            for wav_path in ep["query_paths"]:
                try:
                    w = load_wav(wav_path, target_sr=CFG.sample_rate).to(device)
                    raw_wavs.append(w[0, :max_wav_len])
                except Exception:
                    continue
            if not raw_wavs:
                continue
            valid_eps.append((z_u, raw_wavs))

        if len(valid_eps) < 2:
            continue

        split_rng = random.Random(epoch * 104729 + batch_start)
        shuffled = valid_eps[:]
        split_rng.shuffle(shuffled)
        n_train = max(1, min(len(shuffled) - 1, round(len(shuffled) * meta_split_ratio)))
        meta_train_eps = shuffled[:n_train]
        meta_test_eps = shuffled[n_train:]

        optimizer.zero_grad()

        # Relation loss: standard (non-meta) gradient, no create_graph needed --
        # accumulates directly into .grad via a normal backward.
        loss_relation_val = 0.0
        if len(shuffled) > 1 and lambda_relation > 0.0:
            z_batch_all = torch.stack([z_u for z_u, _ in shuffled])  # [P, 1024]
            deltas_all = generator(z_batch_all)
            z_norm = F.normalize(z_batch_all, dim=1)
            d_norm = F.normalize(deltas_all, dim=1)
            sim_z = z_norm @ z_norm.T
            sim_d = d_norm @ d_norm.T
            off_diag = ~torch.eye(len(shuffled), dtype=torch.bool, device=device)
            loss_relation = F.mse_loss(sim_d[off_diag], sim_z[off_diag])
            (lambda_relation * loss_relation).backward()
            loss_relation_val = loss_relation.item()

        # --- meta-train: cheap per-episode encoder backward -> upstream grads,
        # deltas stay connected to the (small) generator graph only. ---
        n_train = len(meta_train_eps)
        train_deltas, train_upstream_grads = [], []
        loss_train_val = 0.0
        for z_u, raw_wavs in meta_train_eps:
            l, delta, ug = delta_and_upstream_grad(
                z_u, raw_wavs, generator, weights, zonos_enc, campplus,
                yourtts_enc, ecapa, tiler, device, max_wav_len,
            )
            train_deltas.append(delta)
            train_upstream_grads.append(ug / n_train)
            loss_train_val += l / n_train

        # ONE create_graph=True differentiation, confined entirely to the
        # generator's own (tiny) forward passes -- this is the direct
        # d(L_train)/d(theta) term, and also the graph fast_weights needs to
        # stay differentiable w.r.t. theta for the meta-test step below.
        grad_inner = torch.autograd.grad(
            outputs=train_deltas, inputs=params, grad_outputs=train_upstream_grads,
            create_graph=True, allow_unused=True,
        )
        fast_weights = OrderedDict(
            (name, p if g is None else p - meta_inner_lr * g)
            for name, p, g in zip(param_names, params, grad_inner)
        )

        # --- meta-test: same cheap-encoder-backward trick, using fast_weights. ---
        n_test = len(meta_test_eps)
        test_deltas, test_upstream_grads = [], []
        loss_test_val = 0.0
        for z_u, raw_wavs in meta_test_eps:
            l, delta, ug = delta_and_upstream_grad(
                z_u, raw_wavs, generator, weights, zonos_enc, campplus,
                yourtts_enc, ecapa, tiler, device, max_wav_len, fast_weights=fast_weights,
            )
            test_deltas.append(delta)
            test_upstream_grads.append(ug * meta_beta / max(n_test, 1))
            loss_test_val += l / max(n_test, 1)

        # Second differentiation: test_deltas -> fast_weights -> grad_inner's
        # graph -> theta. Still entirely generator-sized (functional_call's
        # forward + the elementwise fast_weights formula + grad_inner's own
        # graph) -- never touches the encoders' graphs a second time.
        if test_deltas:
            grad_outer = torch.autograd.grad(
                outputs=test_deltas, inputs=params, grad_outputs=test_upstream_grads,
                allow_unused=True,
            )
            for p, g in zip(params, grad_outer):
                if g is not None:
                    p.grad = g.detach().clone() if p.grad is None else p.grad + g.detach()

        # Direct L_train(theta) contribution (grad_inner IS d(L_train)/d(theta),
        # just also happens to carry a create_graph=True graph we already
        # consumed above for fast_weights).
        for p, g in zip(params, grad_inner):
            if g is not None:
                gd = g.detach()
                p.grad = gd.clone() if p.grad is None else p.grad + gd

        loss_meta_val = loss_train_val + meta_beta * loss_test_val + lambda_relation * loss_relation_val

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        has_nonfinite = any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in params
        )
        if has_nonfinite:
            optimizer.zero_grad()
            print(f"[epoch {epoch} batch {batch_start}] non-finite grad (MLDG, step skipped)", flush=True)
        else:
            optimizer.step()

        total_loss_val += loss_meta_val
        num_batches += 1
        torch.cuda.empty_cache()

        # Mid-epoch checkpointing -- MLDG epochs are far slower than plain
        # (double-backward), so this environment's recurring SIGSEGVs are
        # even more likely to interrupt an epoch before it finishes.
        CKPT_EVERY_BATCHES = 8
        if num_batches % CKPT_EVERY_BATCHES == 0:
            ckpt = {
                "epoch": epoch - 1,
                "generator": generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": CFG,
            }
            torch.save(ckpt, ckpt_dir / "generator_latest.pt")

    return total_loss_val / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--resume", default=None, help="Resume an in-progress MLDG run (loads epoch+optimizer too)")
    parser.add_argument("--init_from", default=None, help="Warm-start generator weights only from a non-MLDG checkpoint (fresh epoch 0, fresh optimizer)")
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--max_wav_seconds", type=float, default=CFG.max_wav_seconds,
                         help="Override CFG.max_wav_seconds for this run -- MLDG's create_graph=True "
                              "keeps frozen-encoder activation buffers alive for the whole meta-step "
                              "(never freed early like the non-meta trainer does per-speaker), so "
                              "shorter clips directly cut the dominant memory cost.")
    parser.add_argument("--num_query_per_ep", type=int, default=CFG.num_query_per_ep)
    parser.add_argument("--meta_split_ratio", type=float, default=CFG.ensemble4_meta_split_ratio)
    parser.add_argument("--meta_inner_lr", type=float, default=CFG.ensemble4_meta_inner_lr)
    parser.add_argument("--meta_beta", type=float, default=CFG.ensemble4_meta_beta)
    parser.add_argument("--speakers_per_batch", type=int, default=CFG.speakers_per_batch,
                         help="MLDG's no-AMP + create_graph double-backward is far more memory-hungry "
                              "per speaker than the plain ensemble trainer's AMP path; override down "
                              "from the shared CFG.speakers_per_batch=16 if OOM.")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    hang_log_dir = Path(__file__).parent / "log"
    hang_log_dir.mkdir(parents=True, exist_ok=True)
    hang_trace_file = open(hang_log_dir / "hang_trace_ensemble_full4_mldg.log", "a")
    faulthandler.dump_traceback_later(180, repeat=True, file=hang_trace_file, exit=False)

    ckpt_dir = Path(CFG.ckpt_dir).parent / "ensemble_full4_mldg"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Zonos encoder…", flush=True)
    zonos_enc = SpeakerEmbeddingLDA(device=str(device)).to(device).eval()

    print("Loading CAM++ encoder (onnx2torch-traced, CosyVoice weights, CPU)…", flush=True)
    campplus = torch.jit.load(CFG.campplus_pt_cpu_5s_batch4_patched, map_location="cpu").eval()
    for p in campplus.parameters():
        p.requires_grad_(False)

    print("Loading YourTTS encoder…", flush=True)
    yourtts_enc = build_yourtts_encoder(device)

    print("Loading ECAPA encoder…", flush=True)
    from speechbrain.pretrained import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain/ecapa"),
        run_opts={"device": str(device)},
    )

    print("Building dataset…", flush=True)
    dataset = VoxCeleb1EpisodeDataset(
        root=CFG.voxceleb1_root,
        num_speakers=CFG.num_speakers,
        num_support_videos=CFG.num_support_videos,
        num_query_per_ep=args.num_query_per_ep,
        seed=CFG.seed,
        val_ratio=CFG.val_ratio,
    )
    print(f"  {len(dataset)} speakers", flush=True)

    print("Loading z_caches…", flush=True)
    zonos_z_cache = load_z_cache(Path(CFG.cache_dir) / "z_cache.pt", device)
    campplus_z_cache = load_z_cache(Path(CFG.cache_dir) / "campplus_z_cache.pt", device)
    yourtts_z_cache = load_z_cache(Path(CFG.cache_dir) / "yourtts_z_cache.pt", device)
    ecapa_z_cache = load_z_cache(Path(CFG.cache_dir) / "ecapa_z_cache.pt", device)
    print(f"  {len(zonos_z_cache)} Zonos / {len(campplus_z_cache)} CAM++ / "
          f"{len(yourtts_z_cache)} YourTTS / {len(ecapa_z_cache)} ECAPA speakers in cache", flush=True)

    caches = (zonos_z_cache, campplus_z_cache, yourtts_z_cache, ecapa_z_cache)
    dataset.speakers = [s for s in dataset.speakers if all(s in c for c in caches)]
    print(f"  {len(dataset.speakers)} speakers after cache filtering", flush=True)

    tiler = WaveformTiler(
        noise_level=CFG.noise_level,
        target_snr_db=CFG.ensemble4_target_snr_db,
    )

    generator = WaveformUAPGenerator(
        z_dim=Z_DIM,
        hidden_dim=CFG.gen_hidden_dim,
        patch_length=CFG.patch_length_train,
    ).to(device)

    optimizer = optim.Adam(
        generator.parameters(),
        lr=CFG.ensemble4_lr_train,
        weight_decay=CFG.weight_decay,
    )
    max_wav_len = int(CFG.sample_rate * args.max_wav_seconds)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = CFG.ensemble4_lr_train
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed MLDG run from epoch {start_epoch}, lr forced to {CFG.ensemble4_lr_train}", flush=True)
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        print(f"Warm-started generator weights from {args.init_from} "
              f"(non-MLDG epoch {ckpt['epoch']}); starting fresh MLDG epoch 0 with a fresh optimizer", flush=True)

    print(f"Training (ensemble Zonos+CAM+++YourTTS+ECAPA, MLDG, z_dim={Z_DIM}, "
          f"split_ratio={args.meta_split_ratio}, inner_lr={args.meta_inner_lr}, beta={args.meta_beta})…", flush=True)

    log_path = Path(args.log_file) if args.log_file else ckpt_dir.parent / "train_ensemble_full4_mldg_log.txt"

    epoch_iter = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_iter:
        avg_loss = run_epoch_mldg(
            generator=generator,
            zonos_enc=zonos_enc, campplus=campplus, yourtts_enc=yourtts_enc, ecapa=ecapa,
            tiler=tiler,
            dataset=dataset,
            zonos_z_cache=zonos_z_cache, campplus_z_cache=campplus_z_cache,
            yourtts_z_cache=yourtts_z_cache, ecapa_z_cache=ecapa_z_cache,
            optimizer=optimizer,
            device=device,
            speakers_per_batch=args.speakers_per_batch,
            lambda_relation=CFG.ensemble4_lambda_relation,
            meta_split_ratio=args.meta_split_ratio,
            meta_inner_lr=args.meta_inner_lr,
            meta_beta=args.meta_beta,
            max_wav_len=max_wav_len,
            epoch=epoch,
            ckpt_dir=ckpt_dir,
        )

        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")
        msg = f"[Epoch {epoch:03d}] train_loss = {avg_loss:.4f}"

        if (epoch + 1) % CFG.eval_every == 0 or epoch == args.epochs - 1:
            val = run_val(
                generator=generator,
                zonos_enc=zonos_enc, campplus=campplus, yourtts_enc=yourtts_enc, ecapa=ecapa,
                tiler=tiler,
                dataset=dataset,
                zonos_z_cache=zonos_z_cache, campplus_z_cache=campplus_z_cache,
                yourtts_z_cache=yourtts_z_cache, ecapa_z_cache=ecapa_z_cache,
                device=device,
                lambda_perceptual=CFG.lambda_perceptual,
                max_wav_len=max_wav_len,
                num_eval_speakers=CFG.num_eval_speakers,
            )
            msg += (f"  |  val_zonos = {val['zonos']:.4f}  val_campplus = {val['campplus']:.4f}"
                    f"  val_yourtts = {val['yourtts']:.4f}  val_ecapa = {val['ecapa']:.4f}"
                    f"  val_per = {val['per']:.4f}")

        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        ckpt = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": CFG,
        }
        latest_path = ckpt_dir / "generator_latest.pt"
        torch.save(ckpt, latest_path)
        print(f"  → Saved {latest_path}")

        if epoch == args.epochs - 1:
            final_path = ckpt_dir / f"generator_epoch{epoch:03d}.pt"
            torch.save(ckpt, final_path)
            print(f"  → Saved {final_path}")

    faulthandler.cancel_dump_traceback_later()
    hang_trace_file.close()


if __name__ == "__main__":
    main()
