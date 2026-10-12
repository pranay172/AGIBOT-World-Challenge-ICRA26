"""Fine-tune GE-Sim 2.0 (head camera only, V=1) on the challenge train clips (30 fps).

Training samples mirror GeSimV2WorldModel.step() at inference:
  * chunk 0  : memory = the start frame repeated N_PREV=4 times, future = 25 frames from it.
  * chunk c>0: memory = linspace(start, start + 25c - 1, 4) frames, future = the next 25.
    (Teacher forcing: memory uses GT frames where inference uses generated ones.)
Conditioning = trajectory band (2*band-1) + ray map, resized exactly like the model does.

Objective: x0 regression at the distilled 4-step sigmas, matching the distilled sampler
  noisy = (1 - s) * x0 + s * eps;  x0_pred = noisy - s * transformer(...);  loss = |x0_pred - x0|^2
so the fine-tuned model keeps working with the same 4-step sampler. Only the video trunk
runs (the pose expert never feeds the video branch).

usage:
  python train_gesim2.py --train-root WorldModel_Dataset/train --out runs/lora_r64 [--mode lora|full]
  add --resume auto to continue from <out>/last.pt after an interruption (same command line otherwise),
  --patience N to stop after N validations without improvement; the best val checkpoint is <out>/best.pt.
  --band clip draws off-image grippers clipped at the image edge (band_clip.py; inference picks it up from the
  checkpoint args); --grip-weight W weights the x0 loss by 1 + W * gripper mask (gripper_mask());
  --init-lora <ckpt> continues from a previous run's LoRA with a fresh optimizer.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GE-Sim-V2" / "src"))
sys.path.insert(0, str(ROOT))

from band_clip import gripper_uv, render_traj  # noqa: E402
from run_challenge_gesim2 import MODEL_H, MODEL_W, eef_rpy14, load_camera, load_model, load_state  # noqa: E402

N_PREV, CHUNK, FUT_LAT = 4, 25, 7  # 25 frames -> (25 - 1) // 4 + 1 latent frames
LAT_H, LAT_W = MODEL_H // 8, MODEL_W // 8
SIGMAS = [1.0, 0.9375, 0.8333, 0.625]  # DISTILL_SIGMA_SCHEDULE
T_COND = 0.0001 / 1.0001  # sigma_conditioning / (sigma_conditioning + 1) for memory latents
FPS = 30
BAND_MODES = {"orig": "skip", "clip": "clip"}  # --band -> band_clip offframe mode ("orig" == GE-Sim's renderer)
GRIP_RADIUS_M = 0.06  # half-width of the gripper-loss region around the flange -> tool-tip segment


# ---------------------------------------------------------------- data
def list_clips(train_root, holdout_frac, seed=0):
    clips = sorted(p for p in Path(train_root).iterdir()
                   if (p / "head_intrinsic_params.json").exists() and (p / "head_extrinsic_params_aligned.json").exists()
                   and (p / "proprio_stats.h5").exists() and (p / "head_color.mp4").exists())
    episodes = sorted({c.name.rsplit("-", 1)[0] for c in clips})  # <task>-<ep>
    rng = random.Random(seed)
    held = set(rng.sample(episodes, max(1, int(len(episodes) * holdout_frac))))
    train = [c for c in clips if c.name.rsplit("-", 1)[0] not in held]
    val = [c for c in clips if c.name.rsplit("-", 1)[0] in held]
    return train, val


def read_frames(mp4, idx):
    """Decode the requested frame indices (sorted unique decode, any order out)."""
    want = sorted(set(int(i) for i in idx))
    cap = cv2.VideoCapture(str(mp4))
    got, k = {}, 0
    for target in want:
        while k <= target:
            ok, fr = cap.read()
            if not ok:
                raise IndexError(f"{mp4}: frame {target} unreadable")
            if k == target:
                got[k] = fr
            k += 1
    cap.release()
    return [got[int(i)] for i in idx]


def to_model_frames(bgr_list):
    """list of HxWx3 BGR uint8 -> (T, 3, 384, 512) float in [-1, 1] (as GeSimV2WorldModel._preprocess_obs)."""
    x = torch.from_numpy(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_list])).permute(0, 3, 1, 2)
    x = F.interpolate(x.float() / 255.0, (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


def build_cond(eef, K, c2w0, band="orig"):
    """eef (N_PREV + CHUNK, 14) -> (9, N_PREV + FUT_LAT, LAT_H, LAT_W): band*2-1 + rays, resized as in model.py."""
    from gesim.models.gesim_v2.raymap import raymap_from_c2w
    from gesim.models.gesim_v2.utils.geometry_utils import resize_traj_and_ray

    T = eef.shape[0]
    c2w = np.repeat(c2w0[None, None], T, axis=1)  # (V=1, T, 4, 4); camera is static
    traj = render_traj(eef, c2w, K[None], offframe=BAND_MODES[band])[None] * 2.0 - 1.0  # (1, 3, 1, T, H, W)
    rays = raymap_from_c2w(K[None], c2w, res=(MODEL_H, MODEL_W)).float()  # (1, 6, 1, T, H, W)
    cond = resize_traj_and_ray(torch.cat([traj, rays], 1), mem_size=N_PREV, future_size=FUT_LAT,
                               height=LAT_H, width=LAT_W)
    return cond[0, :, 0]


def gripper_mask(end_pos, end_quat, K, c2w0):
    """Soft mask (FUT_LAT, LAT_H, LAT_W) of the gripper regions in the CHUNK future frames: a capsule of
    radius GRIP_RADIUS_M (in pixels at the segment's depth) around each arm's flange -> tool-tip segment,
    evaluated at latent-pixel centres; latent k covers frames 4k-3..4k (latent 0 = frame 0), max over them."""
    g = gripper_uv(end_pos, end_quat, K, c2w0)  # (CHUNK, 2, 2, 3) [flange, tip] x [u, v, depth]
    yy, xx = np.meshgrid((np.arange(LAT_H) + 0.5) * 8, (np.arange(LAT_W) + 0.5) * 8, indexing="ij")
    pix = np.stack([xx, yy], -1)[None, None]  # (1, 1, h, w, 2)
    a, b = g[:, :, 0, None, None, :2], g[:, :, 1, None, None, :2]  # (T, 2, 1, 1, 2)
    ab = b - a
    t = np.clip(((pix - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-6), 0, 1)
    d = np.linalg.norm(pix - (a + t[..., None] * ab), axis=-1)  # (T, 2, h, w)
    depth = np.maximum(g[:, :, :, 2].mean(-1), 0.05)[:, :, None, None]
    r = K[0, 0] * GRIP_RADIUS_M / depth
    m = np.clip(1.0 - (d - r) / 8.0, 0.0, 1.0)  # one-latent-pixel soft edge
    m[g[:, :, :, 2].min(-1) <= 0] = 0.0  # behind the camera
    m = m.max(1)  # both arms: (T, h, w)
    lat = [m[0]] + [m[4 * k - 3:4 * k + 1].max(0) for k in range(1, FUT_LAT)]
    return np.stack(lat).astype(np.float32)


class ClipWindows(Dataset):
    """One random (chunk index, start) window per clip per epoch."""

    def __init__(self, clips, p_chunk0=0.4, max_chunk=8, fixed_seed=None, band="orig", grip_mask=False):
        self.clips, self.p0, self.max_chunk, self.fixed_seed = clips, p_chunk0, max_chunk, fixed_seed
        self.band, self.grip_mask = band, grip_mask

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.fixed_seed is None else self.fixed_seed + i)
        clip = self.clips[i]
        state = load_state(clip / "proprio_stats.h5")
        cap = cv2.VideoCapture(str(clip / "head_color.mp4"))
        n_vid, h, w = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(4)), int(cap.get(3))
        cap.release()
        L = min(state["end_pos"].shape[0], n_vid)
        if L < CHUNK:
            return self[(i + 1) % len(self)]
        c_max = min(self.max_chunk, L // CHUNK - 1)
        c = 0 if (c_max == 0 or rng.random() < self.p0) else int(rng.integers(1, c_max + 1))
        s = int(rng.integers(0, L - CHUNK * (c + 1) + 1))
        if c == 0:
            mem_idx = np.full(N_PREV, s)
        else:
            mem_idx = torch.linspace(s, s + CHUNK * c - 1, N_PREV).long().numpy()  # same rounding as model.py
        fut_idx = s + CHUNK * c + np.arange(CHUNK)
        idx = np.concatenate([mem_idx, fut_idx])

        frames = to_model_frames(read_frames(clip / "head_color.mp4", idx))  # (29, 3, H, W)
        K, c2w0 = load_camera(clip, (h, w))
        cond = build_cond(eef_rpy14(state)[idx], K, c2w0, self.band)
        out = dict(mem=frames[:N_PREV].half(), fut=frames[N_PREV:].permute(1, 0, 2, 3).half(),
                   cond=cond.half(), chunk=c)
        if self.grip_mask:
            out["gmask"] = torch.from_numpy(gripper_mask(state["end_pos"][fut_idx], state["end_quat"][fut_idx],
                                                         K, c2w0)).half()
        return out


# ---------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale


LORA_TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2")


def inject_lora(transformer, r, alpha):
    names = [n for n, m in transformer.named_modules()
             if isinstance(m, nn.Linear) and n.startswith("transformer_blocks.") and n.split(".", 2)[2] in LORA_TARGETS]
    for n in names:
        parent_name, child = n.rsplit(".", 1)
        parent = transformer.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha))
    return names


def lora_state(transformer):
    # names without the CheckpointedBlock ".block" level, so they match the plain inference model
    return {n.replace(".block.", ".", 1): (m.A.detach().cpu(), m.B.detach().cpu(), m.scale)
            for n, m in transformer.named_modules() if isinstance(m, LoRALinear)}


def get_lora_module(transformer, name):
    try:
        return transformer.get_submodule(name)
    except AttributeError:  # block wrapped in CheckpointedBlock
        head, idx, rest = name.split(".", 2)
        return transformer.get_submodule(f"{head}.{idx}.block.{rest}")


def merge_lora(transformer, path):
    """Fold a saved LoRA (or full trunk state) into the transformer weights in place."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("mode") == "full":
        missing, unexpected = transformer.load_state_dict(ckpt["state"], strict=False)
        assert not unexpected, unexpected[:5]
        return len(ckpt["state"])
    with torch.no_grad():
        for n, (A, B, scale) in ckpt["lora"].items():
            lin = transformer.get_submodule(n)
            lin.weight += (scale * (B.float() @ A.float())).to(lin.weight.device, lin.weight.dtype)
    return len(ckpt["lora"])


class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if not self.training:
            return self.block(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(self.block, *args, use_reentrant=False, **kwargs)


# ---------------------------------------------------------------- model step
class Trunk:
    def __init__(self, wm):
        self.tr, self.vae = wm.transformer, wm.vae
        cfg = self.vae.config
        dev = wm.device
        self.mean = torch.tensor(cfg.latents_mean, device=dev).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(cfg.latents_std, device=dev).view(1, -1, 1, 1, 1)
        self.prompt = wm.prompt_embeds  # (1, seq, 1024)
        self.dev, self.dtype = dev, wm.dtype

    @torch.no_grad()
    def encode(self, video):  # (B, 3, T, H, W) in [-1, 1]
        lat = self.vae.encode(video.to(self.dev, self.dtype)).latent_dist.mode().float()
        return (lat - self.mean) / self.std

    def x0_loss(self, batch, sigma_idx=None, gen=None, grip_weight=0.0, parts=False):
        """Per-sample x0 MSE. With grip_weight > 0 (needs batch["gmask"]) the MSE is weighted by
        1 + grip_weight * gripper mask and renormalised by the mean weight, so the loss scale stays put.
        parts=True also returns the plain MSE and the MSE inside the gripper mask (for validation)."""
        B = batch["fut"].shape[0]
        mem = batch["mem"].to(self.dev)  # (B, 4, 3, H, W)
        with torch.no_grad():
            mem_lat = self.encode(mem.flatten(0, 1).unsqueeze(2)).unflatten(0, (B, N_PREV)).squeeze(3)  # (B,4,16,h,w)
            mem_lat = mem_lat.permute(0, 2, 1, 3, 4)  # (B, 16, 4, h, w); each memory frame encoded alone
            x0 = self.encode(batch["fut"])  # (B, 16, 7, h, w)
        if sigma_idx is None:
            sigma_idx = torch.randint(0, len(SIGMAS), (B,), generator=gen)
        sig = torch.tensor([SIGMAS[int(k)] for k in sigma_idx], device=self.dev).view(B, 1, 1, 1, 1)
        eps = torch.randn(x0.shape, generator=gen).to(self.dev) if gen is not None else torch.randn_like(x0)
        noisy = (1 - sig) * x0 + sig * eps

        T_lat = N_PREV + x0.shape[2]
        ind = torch.zeros(1, 1, T_lat, 1, 1, device=self.dev)
        ind[:, :, :N_PREV] = 1.0
        hidden = torch.cat([torch.cat([mem_lat, noisy], 2), batch["cond"].to(self.dev).float()], 1)
        timestep = ind * T_COND + (1 - ind) * sig  # (B, 1, T_lat, 1, 1)
        cond_mask = ind.expand(B, 1, T_lat, LAT_H, LAT_W)
        out = self.tr(
            hidden_states=hidden.to(self.dtype), timestep=timestep.to(self.dtype),
            encoder_hidden_states=self.prompt.expand(B, -1, -1), fps=FPS,
            condition_mask=cond_mask.to(self.dtype),
            padding_mask=torch.zeros(1, 1, LAT_H, LAT_W, device=self.dev, dtype=self.dtype),
            return_dict=False, n_view=1, return_pose=False, compute_video_trunk=True, is_inference=False,
        )[0]["video"][:, :, N_PREV:].float()
        x0_pred = noisy - sig * out
        err = (x0_pred - x0) ** 2
        plain = err.mean(dim=(1, 2, 3, 4))  # (B,)
        if grip_weight > 0 or parts:
            m = batch["gmask"].to(self.dev).float().unsqueeze(1)  # (B, 1, 7, h, w)
        if grip_weight > 0:
            w = 1.0 + grip_weight * m
            per = (err * w).mean(dim=(1, 2, 3, 4)) / w.mean(dim=(1, 2, 3, 4))
        else:
            per = plain
        if not parts:
            return per.mean(), per.detach(), sigma_idx
        grip = (err * m).mean(dim=(1, 2, 3, 4)) / m.mean(dim=(1, 2, 3, 4)).clamp_min(1e-6)  # MSE inside the mask
        return per.mean(), per.detach(), sigma_idx, dict(plain=plain.detach(), grip=grip.detach(),
                                                         has_grip=(m.flatten(1).amax(1) > 0))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, default=ROOT / "WorldModel_Dataset/train")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "weights/gesim2/checkpoints/gesim_community_v2.0.1_g01op_distill_2B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=5000, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--p-chunk0", type=float, default=0.4)
    ap.add_argument("--holdout-frac", type=float, default=0.01)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-n", type=int, default=48)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many validations without val improvement (0 = never)")
    ap.add_argument("--min-delta", type=float, default=0.0, help="val-mean improvement that counts")
    ap.add_argument("--max-clips", type=int, default=0, help="debug: limit train clips")
    ap.add_argument("--keep-blocks", type=int, default=0,
                    help="keep activations of the last N of 28 blocks (no recompute): faster, ~1.5 GB VRAM each")
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' = <out>/last.pt if present")
    ap.add_argument("--init-lora", type=Path, default=None,
                    help="start from this LoRA checkpoint's weights (fresh optimizer/schedule, step 0)")
    ap.add_argument("--band", choices=list(BAND_MODES), default="orig",
                    help="orig = GE-Sim band (skips off-image grippers); clip = draw them clipped at the image edge")
    ap.add_argument("--grip-weight", type=float, default=0.0,
                    help="x0 loss weight 1 + w * gripper mask (0 = plain MSE); val selects on the same weighting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    train_clips, val_clips = list_clips(args.train_root, args.holdout_frac)
    if args.max_clips:
        train_clips = random.Random(0).sample(train_clips, min(args.max_clips, len(train_clips)))
    print(f"train clips {len(train_clips)}, held-out clips {len(val_clips)}", flush=True)

    wm = load_model(args.checkpoint, 4, liger=False)
    tr = wm.transformer
    tr.requires_grad_(False)
    if args.mode == "lora":
        names = inject_lora(tr, args.rank, args.alpha)
        params = [p for n, p in tr.named_parameters() if n.endswith(".A") or n.endswith(".B")]
        print(f"LoRA r={args.rank} on {len(names)} linears, {sum(p.numel() for p in params) / 1e6:.1f}M params",
              flush=True)
    else:
        # full fine-tune of the video trunk; keep weights fp32 master copies via the optimizer on bf16 params
        tr.float()
        params = []
        for n, p in tr.named_parameters():
            if not (n.startswith("pose_") or "video_buffer" in n):
                p.requires_grad_(True); params.append(p)
        print(f"full fine-tune: {sum(p.numel() for p in params) / 1e9:.2f}B params", flush=True)
    assert not (args.mode == "full" and args.keep_blocks), "--keep-blocks is LoRA-only (checkpoint names)"
    if args.keep_blocks <= 0:
        tr.enable_gradient_checkpointing()
    else:  # partial checkpointing: the model-level flag is off, so wrap the blocks we do checkpoint
        n_ck = len(tr.transformer_blocks) - args.keep_blocks
        for i in range(max(0, n_ck)):
            tr.transformer_blocks[i] = CheckpointedBlock(tr.transformer_blocks[i])
        print(f"gradient checkpointing on {max(0, n_ck)} of {len(tr.transformer_blocks)} blocks", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tr.train()
    trunk = Trunk(wm)
    if args.mode == "full":
        trunk.dtype = torch.bfloat16  # inputs; autocast handles the fp32 weights

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    step, best_val, bad_evals = 0, float("inf"), 0
    if args.resume == "auto":
        args.resume = args.out / "last.pt" if (args.out / "last.pt").exists() else None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.mode == "lora":
            for n, (A, B, _) in ck["lora"].items():
                m = get_lora_module(tr, n); m.A.data.copy_(A); m.B.data.copy_(B)
        else:
            tr.load_state_dict(ck["state"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]
        best_val, bad_evals = ck.get("best_val", best_val), ck.get("bad_evals", 0)
        print(f"resumed from {args.resume} at step {step} (best val {best_val:.4f}, bad evals {bad_evals})", flush=True)
    elif args.init_lora:
        assert args.mode == "lora"
        ck = torch.load(args.init_lora, map_location="cpu", weights_only=False)
        assert float(ck["args"]["alpha"]) / int(ck["args"]["rank"]) == args.alpha / args.rank, "LoRA scale mismatch"
        for n, (A, B, _) in ck["lora"].items():
            m = get_lora_module(tr, n); m.A.data.copy_(A); m.B.data.copy_(B)
        print(f"initialised LoRA from {args.init_lora} (its step {ck['step']})", flush=True)

    def save(tag, with_opt=True):
        """Atomic save. Without opt it holds only the weights (enough for inference / merge_lora)."""
        ck = dict(mode=args.mode, step=step, best_val=best_val, bad_evals=bad_evals,
                  args={k: str(v) for k, v in vars(args).items()})
        if with_opt:
            ck.update(opt=opt.state_dict(), sched=sched.state_dict())
        if args.mode == "lora":
            ck["lora"] = lora_state(tr)
        else:
            ck["state"] = {n: p.detach().to(torch.bfloat16).cpu() for n, p in tr.named_parameters() if p.requires_grad}
        tmp = args.out / f".{tag}.pt.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, args.out / f"{tag}.pt")

    val_ds = ClipWindows(val_clips[: args.val_n], p_chunk0=args.p_chunk0, fixed_seed=1234, band=args.band,
                         grip_mask=True)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=min(4, args.workers))

    @torch.no_grad()
    def validate():
        """Plain x0-MSE per sigma (comparable across runs), gripper-region MSE per sigma, and the selection
        metric (the training objective: plain MSE when --grip-weight 0)."""
        tr.eval()
        tot, grip, sel = torch.zeros(len(SIGMAS)), torch.zeros(len(SIGMAS)), torch.zeros(len(SIGMAS))
        cnt, gcnt = 0, 0
        g = torch.Generator().manual_seed(0)
        for b in val_dl:
            for k in range(len(SIGMAS)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, per, _, pt = trunk.x0_loss(b, sigma_idx=torch.tensor([k]), gen=g, grip_weight=args.grip_weight,
                                                  parts=True)
                tot[k] += pt["plain"].sum().cpu(); sel[k] += per.sum().cpu()
                grip[k] += (pt["grip"] * pt["has_grip"]).sum().cpu()
            cnt += 1; gcnt += int(pt["has_grip"].sum())
        tr.train()
        return dict(val=(tot / max(cnt, 1)).tolist(), grip=(grip / max(gcnt, 1)).tolist(),
                    sel=float((sel / max(cnt, 1)).mean()))

    def report(st, r):
        print(f"step {st} val x0-mse per sigma {['%.4f' % x for x in r['val']]} mean {np.mean(r['val']):.4f} | "
              f"grip {['%.4f' % x for x in r['grip']]} mean {np.mean(r['grip']):.4f} | sel {r['sel']:.4f}", flush=True)
        log.write(json.dumps(dict(step=st, **r)) + "\n"); log.flush()

    log = open(args.out / "log.jsonl", "a")
    if step == 0:
        report(0, validate())

    ds = ClipWindows(train_clips, p_chunk0=args.p_chunk0, band=args.band, grip_mask=args.grip_weight > 0)
    t0, step0, ema, ema_s = time.time(), step, None, [None] * len(SIGMAS)
    stop = False
    while step < args.steps and not stop:
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                        persistent_workers=False, prefetch_factor=4)
        it = iter(dl)
        while step < args.steps:
            opt.zero_grad(set_to_none=True)
            try:
                for _ in range(args.accum):
                    b = next(it)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, per, sidx = trunk.x0_loss(b, grip_weight=args.grip_weight)
                    (loss / args.accum).backward()
                    for l, k in zip(per.tolist(), sidx.tolist()):
                        ema_s[k] = l if ema_s[k] is None else 0.98 * ema_s[k] + 0.02 * l
                    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            except StopIteration:
                break  # new epoch (partial accumulation dropped)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 10 == 0:
                el = time.time() - t0
                print(f"step {step} loss_ema {ema:.4f} per_sigma {['%.3f' % (x or 0) for x in ema_s]} "
                      f"gn {gn:.3f} lr {sched.get_last_lr()[0]:.2e} {el / (step - step0):.2f}s/step "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
                log.write(json.dumps(dict(step=step, loss=ema, gn=float(gn))) + "\n"); log.flush()
            if step % args.val_every == 0 or step == args.steps:
                r = validate()
                report(step, r)
                if r["sel"] < best_val - args.min_delta:
                    best_val, bad_evals = r["sel"], 0
                    save("best", with_opt=False)
                    print(f"step {step} new best val {best_val:.4f} -> best.pt", flush=True)
                else:
                    bad_evals += 1
                    if args.patience and bad_evals >= args.patience:
                        print(f"early stop at step {step}: no val improvement in {bad_evals} evals "
                              f"(best {best_val:.4f})", flush=True)
                        stop = True
            if step % args.save_every == 0 or step == args.steps or stop:
                save(f"step{step:06d}", with_opt=False); save("last")
            if stop:
                break
    save("last")
    print("done", flush=True)


if __name__ == "__main__":
    main()
