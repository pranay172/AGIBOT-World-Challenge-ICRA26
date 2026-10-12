"""Validate test-set predictions against the challenge rules and package them as submission.zip.

Rules (challenge page): PREDICTIONS/<task>/<episode>/{0,1,2}/video/frame_%05d.jpg with n_actions - 1 frames each,
PREDICTIONS/meta_info.txt whose first line is the registered team email, zip < 500 MB.

  python eval/make_submission.py --pred preds/test/lora_r64_grip_step001000 --email you@example.com
"""
import argparse
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py

ROOT = Path(__file__).resolve().parents[1]
N_PRED = 3
MAX_BYTES = 500 * 1024**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="<task>/<episode>/<k>/video/frame_*.jpg tree")
    ap.add_argument("--input-root", type=Path, default=ROOT / "WorldModel_Dataset/test/info_dataset")
    ap.add_argument("--email", required=True, help="team registration email (first line of meta_info.txt)")
    ap.add_argument("--out", type=Path, default=ROOT / "submission.zip")
    args = ap.parse_args()

    eps = sorted(p for p in args.input_root.glob("*/*") if p.is_dir())
    errors, n_frames = [], 0
    for ep in eps:
        with h5py.File(ep / "proprio_stats.h5", "r") as f:
            want = f["state/end/position"].shape[0] - 1
        for k in range(N_PRED):
            vid = args.pred / ep.parent.name / ep.name / str(k) / "video"
            names = sorted(p.name for p in vid.glob("*.jpg"))
            if names != [f"frame_{i:05d}.jpg" for i in range(want)]:
                errors.append(f"{vid.relative_to(args.pred)}: {len(names)} frames, want frame_00000..{want - 1:05d}")
                continue
            n_frames += want
            h, w = cv2.imread(str(vid / names[-1])).shape[:2]
            if (h, w) != (480, 640):
                errors.append(f"{vid.relative_to(args.pred)}: {w}x{h}, want 640x480")
    extra = {f"{p.parent.name}/{p.name}" for p in args.pred.glob("*/*") if p.is_dir()} - {
        f"{p.parent.name}/{p.name}" for p in eps}
    if extra:
        errors.append(f"episodes not in the test set: {sorted(extra)[:5]}")
    if errors:
        raise SystemExit("invalid predictions:\n  " + "\n  ".join(errors[:20]))
    print(f"{len(eps)} episodes x {N_PRED} predictions, {n_frames} frames: ok")

    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "meta_info.txt"
        meta.write_text(args.email.strip() + "\n")
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
            z.write(meta, "PREDICTIONS/meta_info.txt")
            for ep in eps:
                for f in sorted((args.pred / ep.parent.name / ep.name).rglob("*.jpg")):
                    z.write(f, "PREDICTIONS/" + f.relative_to(args.pred).as_posix())
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1024**2:.1f} MB)")
    if size >= MAX_BYTES:
        raise SystemExit("submission.zip is over the 500 MB limit")
    with zipfile.ZipFile(args.out) as z:
        assert z.testzip() is None


if __name__ == "__main__":
    main()
