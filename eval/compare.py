"""Print a leaderboard of every scored run: python eval/compare.py [preds/val]"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "preds" / "val")
rows = []
for summary in sorted(root.glob("*_scores/summary.json")):
    m = json.loads(summary.read_text())["overall"]
    rows.append((summary.parent.name.removesuffix("_scores"), m))

print(f"{'run':<32} {'n':>4} {'PSNR':>7} {'SC':>7} {'nDTW':>7} {'score':>7} | {'SC_w':>7} {'score_w':>7}")
for name, m in sorted(rows, key=lambda r: -r[1]["score"]):
    print(f"{name:<32} {m['n_trials']:>4} {m['psnr']:7.3f} {m['scene_consistency']:7.4f} {m['ndtw']:7.4f} "
          f"{m['score']:7.4f} | {m['scene_consistency_w']:7.4f} {m['score_w']:7.4f}")
