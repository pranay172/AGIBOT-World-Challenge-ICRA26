"""Run GE-Sim 2.0 (head camera only) on challenge-format episodes.

Episode dir layout (info_dataset): frame.png, head_intrinsic_params.json,
head_extrinsic_params_aligned.json, proprio_stats.h5 — 5 fps, N actions, N-1 target frames.

GE-Sim 2.0 is trained on 30 fps Genie-01 data with 16-D joint actions
[L7_arm, L_grip(0..1), R7_arm, R_grip(0..1)] plus an EEF "trajectory band" image.
With --fps-mode upsample30 we interpolate the episode to 30 fps, generate, and keep
every 6th frame; native5 feeds the 5 fps sequence directly.

Output: <output-root>/<task>/<episode>/<pred_idx>/video/frame_%05d.jpg at 640x480.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GE-Sim-V2" / "src"))

MODEL_H, MODEL_W = 384, 512
UPSAMPLE = 6  # 5 fps challenge data -> 30 fps training rate
GRIP_MIN, GRIP_RANGE = 35.0, 85.0  # raw OmniPicker 35..120 <-> model 0..1
MIN_BAND_T = 50  # gesim_v2 treats shorter uploaded bands as closed-loop windows
CHUNK = 25  # gesim_v2 chunk size; a final chunk of <= 7 actions trips its conditioning resize


# ---------------------------------------------------------------- episode inputs
def load_state(h5_path):
    with h5py.File(h5_path, "r") as f:
        return {
            "end_pos": np.asarray(f["state/end/position"], dtype=np.float64),  # (T, 2, 3)
            "end_quat": np.asarray(f["state/end/orientation"], dtype=np.float64),  # (T, 2, 4) xyzw
            "joints": np.asarray(f["state/joint/position"], dtype=np.float64),  # (T, 14) [L7, R7]
            "gripper": np.asarray(f["state/effector/position"], dtype=np.float64),  # (T, 2) raw
        }


def upsample_state(state, factor):
    """Interpolate to `factor`x rate; original step k lands on index k*factor."""
    n = state["end_pos"].shape[0]
    src = np.arange(n) * factor
    dst = np.arange((n - 1) * factor + 1)
    out = {}
    for key, v in state.items():
        if key == "end_quat":
            out[key] = np.stack(
                [Slerp(src, Rotation.from_quat(v[:, a]))(dst).as_quat() for a in range(v.shape[1])], axis=1
            )
        elif key in ("end_pos", "joints") and n >= 3:
            out[key] = CubicSpline(src, v, axis=0)(dst)
        else:
            flat = v.reshape(n, -1)
            out[key] = np.stack([np.interp(dst, src, flat[:, k]) for k in range(flat.shape[1])], 1).reshape(
                (len(dst),) + v.shape[1:]
            )
    return out


def grip01(raw):
    return np.clip((raw - GRIP_MIN) / GRIP_RANGE, 0.0, 1.0)


def actions16(state):
    """(T, 16) [L7, L_grip, R7, R_grip] with grippers in 0..1."""
    j, g = state["joints"], grip01(state["gripper"])
    return np.concatenate([j[:, :7], g[:, :1], j[:, 7:14], g[:, 1:2]], axis=1).astype(np.float32)


def eef_rpy14(state):
    """(T, 14) [L_xyz, L_rpy, L_grip, R_xyz, R_rpy, R_grip] for the trajectory band."""
    parts = []
    g = grip01(state["gripper"])
    for arm in range(2):
        rpy = Rotation.from_quat(state["end_quat"][:, arm]).as_euler("xyz")
        parts += [state["end_pos"][:, arm], rpy, g[:, arm : arm + 1]]
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_camera(ep_dir, orig_hw):
    info = json.loads((ep_dir / "head_intrinsic_params.json").read_text())["intrinsic"]
    sx, sy = MODEL_W / orig_hw[1], MODEL_H / orig_hw[0]
    K = np.array([[info["fx"] * sx, 0, info["ppx"] * sx], [0, info["fy"] * sy, info["ppy"] * sy], [0, 0, 1]],
                 dtype=np.float32)
    ext = json.loads((ep_dir / "head_extrinsic_params_aligned.json").read_text())[0]["extrinsic"]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(ext["rotation_matrix"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(ext["translation_vector"], dtype=np.float32)
    return K, c2w  # camera is static in this dataset, so one c2w serves every frame


def build_inputs(ep_dir, fps_mode, band="orig"):
    """Everything GE-Sim 2.0 needs for one episode (head view only, V=1).
    band: "orig" = GE-Sim's renderer; "clip" = off-image grippers drawn clipped at the edge (band_clip.py)."""
    from band_clip import render_traj

    frame_bgr = cv2.imread(str(ep_dir / "frame.png"))
    orig_hw = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    first_frame = np.transpose(rgb, (2, 0, 1))[:, None]  # (3, V=1, H, W) in [0, 1]

    state = load_state(ep_dir / "proprio_stats.h5")
    n_target = state["end_pos"].shape[0] - 1
    stride = UPSAMPLE if fps_mode == "upsample30" else 1
    if stride > 1:
        state = upsample_state(state, stride)

    K, c2w0 = load_camera(ep_dir, orig_hw)
    acts, eef = actions16(state), eef_rpy14(state)
    # Pad to whole chunks (robot holds its last pose), and the band to >= MIN_BAND_T.
    T_act = math.ceil(acts.shape[0] / CHUNK) * CHUNK
    acts = np.concatenate([acts, np.repeat(acts[-1:], T_act - acts.shape[0], 0)])
    T_band = max(T_act, MIN_BAND_T)
    eef_band = np.concatenate([eef, np.repeat(eef[-1:], T_band - eef.shape[0], 0)])
    c2w = np.repeat(c2w0[None, None], T_band, axis=1)  # (V=1, T, 4, 4)
    offframe = {"orig": "skip", "clip": "clip"}[band]
    band = render_traj(eef_band, c2w, K[None], offframe=offframe)  # (3, 1, T, H, W) in [0, 1]
    return dict(first_frame=first_frame, K=K[None], c2w=c2w, band=band.numpy(), actions=acts,
                n_target=n_target, stride=stride, orig_hw=orig_hw)


# ---------------------------------------------------------------- generation
def load_model(ckpt_dir, steps, sparge=False, triton_rope=False, liger=True):
    import gesim.models.gesim_v2.model as gm

    gm.N_VIEW = 1  # head camera only; the pipeline reads this module constant at call time
    cfg = dict(checkpoint=str(ckpt_dir), device="cuda", dtype="bfloat16", num_inference_steps=steps,
               distilled_sampling=True, distill_sigma_schedule=gm.DISTILL_SIGMA_SCHEDULE,
               liger_norm=liger, liger_layernorm=liger, triton_rope=triton_rope, sparge_attention=sparge)
    return gm.GeSimV2WorldModel.from_config(cfg)


def generate(model, inp):
    model.reset()
    model.set_camera_params(inp["K"], inp["c2w"][:, 0])
    model.set_episode_data(inp["first_frame"])
    model.set_episode_traj(inp["band"], inp["c2w"])
    acts = inp["actions"]
    frames = []
    for s in range(0, acts.shape[0], model.chunk_size):
        frames.append(model.step(acts[s : s + model.chunk_size]).frames)  # (L, 3, V, H, W) in [0, 1]
    return np.concatenate(frames)[:, :, 0]  # head view: (T, 3, H, W); frame k <-> step k


def save_frames(gen, inp, out_dir, jpg_quality):
    # gen[k] corresponds to (upsampled) step k; step 0 is the given first frame.
    idx = [k * inp["stride"] for k in range(1, inp["n_target"] + 1)]
    idx = [min(i, gen.shape[0] - 1) for i in idx]
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = inp["orig_hw"]
    for j, i in enumerate(idx):
        img = (np.clip(gen[i].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out_dir / f"frame_{j:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "weights/gesim2/checkpoints/gesim_community_v2.0.1_g01op_distill_2B")
    ap.add_argument("--fps-mode", choices=["native5", "upsample30"], default="upsample30")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n-pred", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes-file", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--triton-rope", action="store_true")
    ap.add_argument("--lora", type=Path, default=None, help="train_gesim2.py checkpoint to merge")
    ap.add_argument("--band", choices=["orig", "clip"], default=None,
                    help="trajectory band renderer; default: the one the --lora checkpoint was trained with, else orig")
    args = ap.parse_args()

    import torch

    if args.lora:
        trained = torch.load(args.lora, map_location="cpu", weights_only=False).get("args", {}).get("band", "orig")
        if args.band is None:
            args.band = trained
        elif args.band != trained:
            print(f"WARNING: --band {args.band} but {args.lora} was trained with --band {trained}", flush=True)
    args.band = args.band or "orig"

    eps = sorted(p for p in args.input_root.glob("*/*") if p.is_dir())
    if args.episodes_file:
        wanted = {l.strip() for l in args.episodes_file.read_text().splitlines() if l.strip()}
        eps = [p for p in eps if f"{p.parent.name}/{p.name}" in wanted]
    print(f"{len(eps)} episodes, fps_mode={args.fps_mode}, steps={args.steps}, n_pred={args.n_pred}, "
          f"band={args.band}", flush=True)

    model = load_model(args.checkpoint, args.steps, triton_rope=args.triton_rope)
    if args.lora:
        from train_gesim2 import merge_lora

        print(f"merged {merge_lora(model.transformer, args.lora)} tensors from {args.lora}", flush=True)
    t_start = time.time()
    for n, ep in enumerate(eps):
        inp = None
        for k in range(args.n_pred):
            out_dir = args.output_root / ep.parent.name / ep.name / str(k) / "video"
            if args.skip_existing and out_dir.is_dir() and len(list(out_dir.glob("*.jpg"))) > 0:
                continue
            inp = inp or build_inputs(ep, args.fps_mode, args.band)
            seed = args.seed + k
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            t0 = time.time()
            gen = generate(model, inp)
            save_frames(gen, inp, out_dir, args.jpg_quality)
            print(f"[{n + 1}/{len(eps)}] {ep.parent.name}/{ep.name} pred={k} steps30={inp['actions'].shape[0]} "
                  f"{time.time() - t0:.1f}s (elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
