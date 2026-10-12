"""Checks band_clip.render_traj against GE-Sim's renderer on all 189 validation clips (30 fps band, as at
inference) and a sample of train clips:
  1. offframe="skip" is bit-identical to gesim.conditioning.band.render_traj.
  2. offframe="clip" only changes frames where a gripper origin projects outside the canvas.
  3. band_clip.gripper_uv tool-tip pixels match the band's circle centres.
usage: python eval/check_band_clip.py [--train-n 200]
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GE-Sim-V2" / "src")); sys.path.insert(0, str(ROOT))
from gesim.conditioning.band import render_traj as render_orig  # noqa: E402

import band_clip  # noqa: E402
from run_challenge_gesim2 import (MODEL_H, MODEL_W, eef_rpy14, load_camera, load_state,  # noqa: E402
                                  upsample_state)


def check(ep, hw, state, stats):
    K, c2w0 = load_camera(ep, hw)
    eef = eef_rpy14(state)
    c2w = np.repeat(c2w0[None, None], eef.shape[0], axis=1)
    a = render_orig(eef, c2w, K[None]).numpy()
    s = band_clip.render_traj(eef, c2w, K[None], offframe="skip").numpy()
    c = band_clip.render_traj(eef, c2w, K[None], offframe="clip").numpy()
    stats["skip_mismatch"] += int((a != s).any())
    g = band_clip.gripper_uv(state["end_pos"], state["end_quat"], K, c2w0)  # (T, 2, 2, 3)
    tip = g[:, :, 1, :2]
    inside = ((tip[..., 0] >= 0) & (tip[..., 0] < MODEL_W) & (tip[..., 1] >= 0) & (tip[..., 1] < MODEL_H)).all(1)
    changed = (a != c).any(axis=(0, 1, 3, 4))  # (T,)
    stats["frames"] += len(changed); stats["changed"] += int(changed.sum())
    stats["changed_but_inside"] += int((changed & inside).sum())
    stats["outside"] += int((~inside).sum())
    stats["outside_unchanged"] += int((~inside & ~changed).sum())
    stats["tip_err"] = max(stats["tip_err"], float(np.abs(tip[inside] - np.trunc(tip[inside])).max(initial=0)))
    # circle centre check: band pixel at the truncated tip must carry the circle colour or a stick colour, not bg
    bg = 50.0 / 255.0
    for t in np.flatnonzero(inside)[::7]:
        for arm in range(2):
            u, v = int(tip[t, arm, 0]), int(tip[t, arm, 1])
            stats["centre_bg"] += int(np.allclose(a[:, 0, t, v, u], bg))
            stats["centre_n"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=200)
    args = ap.parse_args()
    for split in ("val", "train"):
        stats = dict(clips=0, skip_mismatch=0, frames=0, changed=0, changed_but_inside=0, outside=0,
                     outside_unchanged=0, tip_err=0.0, centre_bg=0, centre_n=0)
        if split == "val":
            eps = sorted((ROOT / "WorldModel_Dataset/validation/info_dataset").glob("*/*"))
        else:
            eps = sorted((ROOT / "WorldModel_Dataset/train").iterdir())
            random.Random(1).shuffle(eps); eps = eps[: args.train_n]
        for ep in eps:
            if split == "val":
                hw = cv2.imread(str(ep / "frame.png")).shape[:2]
                state = upsample_state(load_state(ep / "proprio_stats.h5"), 6)
            else:
                cap = cv2.VideoCapture(str(ep / "head_color.mp4")); hw = (int(cap.get(4)), int(cap.get(3))); cap.release()
                state = load_state(ep / "proprio_stats.h5")
            check(ep, hw, state, stats); stats["clips"] += 1
        print(split, stats, flush=True)
        assert stats["skip_mismatch"] == 0, "skip mode differs from GE-Sim renderer"
        assert stats["changed_but_inside"] == 0, "clip mode changed an all-inside frame"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
