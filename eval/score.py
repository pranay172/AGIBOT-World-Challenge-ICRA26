"""Local re-implementation of the ICRA 2026 World Model track scoring.

Uses EWMBench's own metric code unchanged (PSNR, DINOv2 scene consistency,
YOLO-World gripper nDTW) without importing its heavy semantics/diversity
modules. Final score = mean(clip(PSNR, 0, 35) / 35, scene_consistency, nDTW).

Usage:
    python eval/score.py --pred PRED_ROOT [--gt GT_ROOT] [--out OUT_DIR]

PRED_ROOT layout: <task>/<episode>/<trial>/video/frame_*.jpg (the submission
layout, without meta_info.txt). Frames are resized in place to 640x480, and
gripper_detection/ + traj/ folders are written next to each video, exactly as
EWMBench's processing.sh does.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EWM = ROOT / "EWMBench"
DEFAULT_GT = ROOT / "WorldModel_Dataset" / "validation" / "gt_dataset"

# Register EWMBench as a bare package so its submodules import without
# running EWMBench/__init__.py (which pulls in Qwen-VL, CLIP, etc.).
_pkg = types.ModuleType("EWMBench")
_pkg.__path__ = [str(EWM / "EWMBench")]
sys.modules["EWMBench"] = _pkg

from EWMBench.basic_metrics import compute_basic_metrics  # noqa: E402
from EWMBench.trajectory_consistency import compute_trajectory_consistency  # noqa: E402
from EWMBench import scene_consistency as sc_mod  # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


video_resize = _load_module("ewm_video_resize", EWM / "processing" / "video_resize.py")
detection = _load_module("ewm_detection_tracking", EWM / "processing" / "detection_tracking.py")


def iter_trials(pred_root):
    for task in sorted(os.listdir(pred_root)):
        task_dir = pred_root / task
        if not task_dir.is_dir():
            continue
        for ep in sorted(os.listdir(task_dir)):
            ep_dir = task_dir / ep
            if not ep_dir.is_dir():
                continue
            for gid in sorted(os.listdir(ep_dir)):
                if (ep_dir / gid / "video").is_dir():
                    yield task, ep, gid


def check_frame_counts(pred_root, gt_root):
    problems = []
    for task, ep, gid in iter_trials(pred_root):
        n_gt = len(list((gt_root / task / ep / "video").glob("frame_*.png")))
        n_pd = len([p for p in (pred_root / task / ep / gid / "video").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if n_gt == 0:
            problems.append(f"{task}/{ep}: no GT frames")
        elif n_pd != n_gt:
            problems.append(f"{task}/{ep}/{gid}: {n_pd} pred frames vs {n_gt} GT")
    return problems


def run_detection(pred_root, gt_root, yolo_ckpt):
    episodes = sorted({(t, e) for t, e, _ in iter_trials(pred_root)})
    for i, (task, ep) in enumerate(episodes):
        gt_ep = gt_root / task / ep
        if not (gt_ep / "traj" / "traj.npy").exists():
            detection.process_video_with_tracking(
                input_path=str(gt_ep / "video"), output_path=str(gt_ep),
                gid=None, model_path=str(yolo_ckpt), data_type="gt")
        ep_dir = pred_root / task / ep
        for gid in sorted(os.listdir(ep_dir)):
            if (ep_dir / gid / "video").is_dir():
                detection.process_video_with_tracking(
                    input_path=str(ep_dir / gid / "video"), output_path=str(ep_dir),
                    gid=gid, model_path=str(yolo_ckpt), data_type="val")
        print(f"[detect] {i + 1}/{len(episodes)} {task}/{ep}", flush=True)


def load_dino(ckpt, mode="weights"):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(EWM / "dino_config.yaml")
    model, _ = sc_mod.build_model_from_cfg(cfg, only_teacher=True)
    model = model.cuda().eval()
    own = model.state_dict()
    # The checkpoint stores a few numpy scalars (training metadata); allow just those.
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if mode == "weights":
        state = state.get("model", state)
    # mode == "official_code": EWMBench iterates the top-level dict (keys model/optimizer/
    # iteration), so no teacher.* keys match and DINOv2 keeps its random init.
    to_load = {}
    for k, v in state.items():
        if k.startswith("teacher"):
            k2 = k.replace("teacher.", "").replace("backbone.", "")
            if k2 in own:
                to_load[k2] = v
    res = model.load_state_dict(to_load, strict=False)
    print(f"[dino:{mode}] loaded {len(to_load)} tensors, {len(res.missing_keys)} missing")
    return model


@torch.no_grad()
def scene_consistency_batched(model, video_dir, device, batch=8):
    """Same math as EWMBench scene_consistency(), with frames batched through DINOv2.

    Note EWMBench takes the cosine over dim=1 of (1, n_patches, dim) features, i.e. across
    patches per channel, then averages over channels; replicated here exactly.
    """
    images = sc_mod.load_video(video_dir)
    _, _, H, W = images.shape
    side = max(H, W)
    pt, pl = (side - H) // 2, (side - W) // 2
    images = torch.nn.functional.pad(images, (pl, side - W - pl, pt, side - H - pt))
    images = sc_mod.dino_transform(518)(images)
    feats = []
    for i in range(0, len(images), batch):
        f = model.forward_features(images[i:i + batch].to(device))["x_norm_patchtokens"]
        feats.append(torch.nn.functional.normalize(f, dim=-1, p=2))
    feats = torch.cat(feats)  # (T, n_patches, dim)
    cos = torch.nn.functional.cosine_similarity
    sim_pre = cos(feats[:-1], feats[1:], dim=1).mean(-1).clamp(min=0)
    sim_fir = cos(feats[:1].expand_as(feats[1:]), feats[1:], dim=1).mean(-1).clamp(min=0)
    return float(((sim_pre + sim_fir) / 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--out", type=Path, default=None, help="Defaults to <pred>/../<pred name>_scores")
    ap.add_argument("--dino_ckpt", type=Path, default=EWM / "ckpt" / "dinov2-EWMBench-v0.1.pth")
    ap.add_argument("--yolo_ckpt", type=Path, default=EWM / "ckpt" / "yoloworld-EWMBench-v0.1.pt")
    ap.add_argument("--skip_detection", action="store_true", help="Reuse existing traj/traj.npy files")
    args = ap.parse_args()

    pred_root, gt_root = args.pred.resolve(), args.gt.resolve()
    out_dir = args.out or pred_root.parent / f"{pred_root.name}_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = check_frame_counts(pred_root, gt_root)
    if problems:
        print(f"WARNING: {len(problems)} frame-count problems (missing frames score 0 PSNR):")
        print("\n".join("  " + p for p in problems[:20]))

    print("[resize] to 640x480 (in place, like EWMBench processing.sh)")
    for task, ep, gid in iter_trials(pred_root):
        video_resize.resize_and_save_images(str(pred_root / task / ep / gid / "video"))

    if not args.skip_detection:
        run_detection(pred_root, gt_root, args.yolo_ckpt)

    print("[psnr]")
    psnr = compute_basic_metrics(str(gt_root), str(pred_root), metric_names=["psnr"])["psnr"]

    print("[ndtw]")
    traj = compute_trajectory_consistency(gt_path=str(gt_root), data_base=str(pred_root))

    video_list = [(t, e, g) for t, e, g in iter_trials(pred_root)]
    sc = {}
    for mode in ("official_code", "weights"):
        print(f"[scene consistency: {mode}]")
        torch.manual_seed(0)  # makes the official_code (random-init) variant reproducible
        dino = load_dino(args.dino_ckpt, mode)
        for i, (t, e, g) in enumerate(video_list):
            sc[(mode, t, e, g)] = scene_consistency_batched(dino, str(pred_root / t / e / g / "video"), "cuda")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(video_list)}", flush=True)
        del dino
        torch.cuda.empty_cache()

    rows = []
    for t, e, g in iter_trials(pred_root):
        p = float(psnr[t][e][g])
        rows.append({
            "task_id": t, "episode_id": e, "trial_id": g,
            "psnr": p,
            "scene_consistency": sc[("official_code", t, e, g)],
            "scene_consistency_w": sc[("weights", t, e, g)],
            "ndtw": float(traj[t][e][g]["ndtw"]),
            "hsd": float(traj[t][e][g]["hsd"]),
            "dyn": float(traj[t][e][g]["dyn"]),
        })

    def summarize(rs):
        keys = ("psnr", "scene_consistency", "scene_consistency_w", "ndtw", "hsd", "dyn")
        m = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        m["psnr_norm"] = float(np.clip(m["psnr"], 0, 35) / 35)
        m["score"] = float(np.mean([m["psnr_norm"], m["scene_consistency"], m["ndtw"]]))
        m["score_w"] = float(np.mean([m["psnr_norm"], m["scene_consistency_w"], m["ndtw"]]))
        m["n_trials"] = len(rs)
        return m

    overall = summarize(rows)
    per_task = {t: summarize([r for r in rows if r["task_id"] == t]) for t in sorted({r["task_id"] for r in rows})}

    with open(out_dir / "per_trial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"overall": overall, "per_task": per_task, "frame_count_problems": problems}, f, indent=2)

    hdr = f"{'task':>8} {'n':>4} {'PSNR':>7} {'SC':>7} {'nDTW':>7} {'score':>7} | {'SC_w':>7} {'score_w':>7}"
    print("\nSC / score: DINOv2 as EWMBench's code loads it (random init; matches EVAC's published SC)")
    print("SC_w / score_w: DINOv2 with the finetuned weights actually loaded\n" + hdr)
    for t, m in list(per_task.items()) + [("ALL", overall)]:
        print(f"{t:>8} {m['n_trials']:>4} {m['psnr']:7.3f} {m['scene_consistency']:7.4f} {m['ndtw']:7.4f} "
              f"{m['score']:7.4f} | {m['scene_consistency_w']:7.4f} {m['score_w']:7.4f}")
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
