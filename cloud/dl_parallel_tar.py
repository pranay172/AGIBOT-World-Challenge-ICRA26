"""Parallel HTTP range download of a .tar.gz, streamed into `tar xz` in order.

Chunks are fetched by N curl workers into a spool dir, piped to tar in sequence and
deleted once consumed, so disk use stays at ~window * chunk_size beyond the extracted data.

usage: python dl_parallel_tar.py URL DEST_DIR [--strip N] [--workers 16] [--chunk-mb 128] [--window 48]
"""
import argparse
import subprocess
import threading
import time
from pathlib import Path


def content_length(url):
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True, check=True).stdout
    sizes = [int(l.split(":", 1)[1]) for l in out.lower().splitlines() if l.startswith("content-length:")]
    return sizes[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--strip", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=128)
    ap.add_argument("--window", type=int, default=48, help="max chunks downloaded ahead of tar")
    ap.add_argument("--spool", type=Path, default=None)
    args = ap.parse_args()

    total = content_length(args.url)
    cs = args.chunk_mb << 20
    n_chunks = (total + cs - 1) // cs
    spool = args.spool or args.dest.parent / (args.dest.name + ".spool")
    spool.mkdir(parents=True, exist_ok=True)
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"{total / 1e9:.2f} GB in {n_chunks} chunks of {args.chunk_mb} MB, {args.workers} workers", flush=True)

    done = [False] * n_chunks
    consumed = [0]  # chunks handed to tar so far
    nxt = [0]
    lock = threading.Condition()

    def worker():
        while True:
            with lock:
                while nxt[0] < n_chunks and nxt[0] >= consumed[0] + args.window:
                    lock.wait()
                if nxt[0] >= n_chunks:
                    return
                i = nxt[0]; nxt[0] += 1
            lo, hi = i * cs, min(total, (i + 1) * cs) - 1
            part, tmp = spool / f"{i:06d}", spool / f"{i:06d}.tmp"
            for attempt in range(20):
                r = subprocess.run(["curl", "-sSL", "--fail", "-r", f"{lo}-{hi}", "-o", str(tmp), args.url])
                if r.returncode == 0 and tmp.stat().st_size == hi - lo + 1:
                    tmp.rename(part)
                    break
                time.sleep(min(60, 2 ** attempt))
            else:
                raise SystemExit(f"chunk {i} failed")
            with lock:
                done[i] = True
                lock.notify_all()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    tar = subprocess.Popen(["tar", "xz", f"--strip-components={args.strip}", "-C", str(args.dest)],
                           stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(n_chunks):
        with lock:
            while not done[i]:
                lock.wait(timeout=30)
        part = spool / f"{i:06d}"
        with open(part, "rb") as f:
            while buf := f.read(8 << 20):
                tar.stdin.write(buf)
        part.unlink()
        with lock:
            consumed[0] = i + 1
            lock.notify_all()
        if i % 10 == 0 or i == n_chunks - 1:
            gb = (i + 1) * cs / 1e9
            rate = gb * 1e3 / (time.time() - t0)
            eta = (total / 1e9 - gb) * 1e3 / max(rate, 1e-6) / 60
            print(f"chunk {i + 1}/{n_chunks}  {gb:.1f} GB  {rate:.1f} MB/s  eta {eta:.0f} min", flush=True)
    tar.stdin.close()
    rc = tar.wait()
    spool.rmdir()
    print(f"tar exit={rc}", flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
