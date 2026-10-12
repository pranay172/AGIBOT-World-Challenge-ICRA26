"""Build reference predictions on the validation set for sanity-checking eval/score.py.

  gt_jpg : GT frames re-saved as JPEG (upper bound; checks frame alignment)
  static : the initial frame repeated for every target frame (trivial baseline)
"""
import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "WorldModel_Dataset" / "validation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gt_jpg", "static"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jpg_quality", type=int, default=95)
    args = ap.parse_args()
    out = args.out or ROOT / "preds" / "val" / args.kind

    for ep_dir in sorted((VAL / "gt_dataset").glob("*/*")):
        task, ep = ep_dir.parent.name, ep_dir.name
        gt_frames = sorted((ep_dir / "video").glob("frame_*.png"))
        init = cv2.imread(str(VAL / "info_dataset" / task / ep / "frame.png"))
        vdir = out / task / ep / "0" / "video"
        vdir.mkdir(parents=True, exist_ok=True)
        for i, gt in enumerate(gt_frames):
            img = cv2.imread(str(gt)) if args.kind == "gt_jpg" else init
            cv2.imwrite(str(vdir / f"frame_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
