"""Render the README demos from the final model's predictions.

docs/demo_val.gif   2 validation episodes, one per row: the real recorded video next to the generated one. Both start from the
                    same single real frame (the only image the model gets) and follow the same recorded actions.
docs/demo_test.gif  4 test episodes in a 2x2 grid (no ground truth released): input frame, then generated video.

Episodes were picked by eye as clear task completions; the captions describe what happens in the video, they are
not official task names. Frames are shown at the challenge's 5 fps.

  python eval/make_demo_gif.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PRED = "lora_r64_grip_step001000"
TEST_CLIPS = [  # (task/episode, prediction seed index, caption)
    ("2014/12394960026700475", 1, "toast: toaster -> plate"),
    ("2019/12394460136001561", 1, "put dumplings in the freezer"),
    ("2022/12399220080501095", 2, "detergent bottle -> box"),
    ("2012/12385780013200460", 0, "wipe the table with a towel"),
]
VAL_CLIPS = [  # (task/episode, caption); validation predictions use seed index 0
    ("543/798749001", "put the pouch in the box"),
    ("497/773025006", "take a package out of the freezer"),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
INPUT_COLOR, REAL_COLOR, GEN_COLOR = (120, 220, 255), (255, 200, 120), (160, 255, 160)  # BGR
HOLD_INPUT, HOLD_END = 4, 5  # GIF frames showing the input frame first, and the last frame before looping


def fit(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def tag(img, left, right, color, bar=20, scale=0.42):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], bar), (0, 0, 0), -1)
    cv2.putText(img, left, (5, bar - 6), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    tw = cv2.getTextSize(right, FONT, scale, 1)[0][0]
    cv2.putText(img, right, (img.shape[1] - 5 - tw, bar - 6), FONT, scale, color, 1, cv2.LINE_AA)
    return img


def footer(grid, text):
    grid = cv2.copyMakeBorder(grid, 0, 20, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(grid, text, (5, grid.shape[0] - 6), FONT, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    return grid


def play(seqs):
    """Hold each tile's last frame so that all tiles start together and the GIF loops cleanly."""
    n = max(map(len, seqs)) + HOLD_END
    return [[s[min(t, len(s) - 1)] for s in seqs] for t in range(n)]


def save_gif(frames, out):
    # a palette per frame: several different scenes don't fit one shared 256-colour palette without speckle
    q = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=200, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


def real_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def val_demo(args, w=320, h=240):
    small = dict(bar=18, scale=0.4)
    seqs = []
    for ep, text in VAL_CLIPS:
        task, e = ep.split("/")
        gen_dir = next(d for d in (args.pred_root / "val").glob(f"{PRED}_*/{ep}/0/video") if d.is_dir())
        gen = sorted(gen_dir.glob("*.jpg"))
        real = real_video(args.data / "val_raw" / f"{task}-{e[:-3]}-{e[-3:]}" / "head_color.mp4")
        assert len(real) == len(gen) + 1, (ep, len(real), len(gen))  # real[0] is the input frame
        first = fit(real[0], w, h)
        seq = [np.hstack([tag(first, "REAL", "input frame", INPUT_COLOR, **small),
                          tag(first, "GENERATED", "input frame", INPUT_COLOR, **small)])] * HOLD_INPUT
        for i, g in enumerate(gen):
            seq.append(np.hstack([
                tag(fit(real[i + 1], w, h), "REAL", f"recorded {i + 1}/{len(gen)}", REAL_COLOR, **small),
                tag(fit(cv2.imread(str(g)), w, h), "GENERATED", f"ours {i + 1}/{len(gen)}", GEN_COLOR, **small)]))
        cap = np.zeros((20, 2 * w, 3), np.uint8)
        cv2.putText(cap, text, (5, 14), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        seqs.append([np.vstack([cap, s, np.zeros((4, 2 * w, 3), np.uint8)]) for s in seq])
    frames = []
    for tiles in play(seqs):
        grid = np.vstack(tiles)  # one clip per row: real | generated
        frames.append(footer(grid, "Validation: real recording vs. ours (GE-Sim 2.0 + LoRA), same first frame + "
                                   "robot actions | AgiBot World, 5 fps"))
    save_gif(frames, args.out_dir / "demo_val.gif")


def test_demo(args, w=320, h=240):
    seqs = []
    for ep, k, text in TEST_CLIPS:
        gen = sorted((args.pred_root / "test" / PRED / ep / str(k) / "video").glob("*.jpg"))
        first = fit(cv2.imread(str(args.data / "test/info_dataset" / ep / "frame.png")), w, h)
        seqs.append([tag(first, text, "real input frame", INPUT_COLOR)] * HOLD_INPUT
                    + [tag(fit(cv2.imread(str(f)), w, h), text, f"generated {i + 1}/{len(gen)}", GEN_COLOR)
                       for i, f in enumerate(gen)])
    frames = [footer(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]),
                     "Test set (no ground truth): ours (GE-Sim 2.0 + LoRA), one real frame + robot actions -> video"
                     " | AgiBot World, 5 fps") for tiles in play(seqs)]
    save_gif(frames, args.out_dir / "demo_test.gif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, default=ROOT / "preds")
    ap.add_argument("--data", type=Path, default=ROOT / "WorldModel_Dataset")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    args = ap.parse_args()
    val_demo(args)
    test_demo(args)


if __name__ == "__main__":
    main()
