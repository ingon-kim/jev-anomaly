"""Jev-Omni image anomaly pilot eval.

data/normal/*, data/anomaly/* -> P(Anomalous) per image -> CSV + metrics.
Needs llama-server running (see restart_server.sh; --no-mmproj-offload required).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "jev-omni-q4"))
from jev_omni_gguf_decide import decide  # reuse adapter in-process, no subprocess per image

OPTIONS = ["Normal", "Anomalous"]  # order matters: decision head weights are positional
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def auroc(labels, scores):
    # Mann-Whitney U with average ranks for ties
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):
        tie = scores == s
        ranks[tie] = ranks[tie].mean()
    n1, n0 = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binary_metrics(labels, scores, thr):
    labels, pred = np.asarray(labels), (np.asarray(scores) >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(thr), "accuracy": float((pred == labels).mean()), "precision": prec, "recall": rec, "f1": f1}


def best_f1(labels, scores):
    return max((binary_metrics(labels, scores, t) for t in np.unique(scores)), key=lambda m: m["f1"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--state", default="The image shows an object photographed for industrial visual inspection.")
    p.add_argument("--question", default="Is the object in the image normal, or does it have a defect or anomaly?")
    p.add_argument("--server", default="http://127.0.0.1:8080")
    p.add_argument("--head", type=Path, default=Path(__file__).parent / "jev-omni-q4" / "decision-head-f32.npz")
    p.add_argument("--out", type=Path, default=Path("results.csv"))
    p.add_argument("--limit", type=int, help="max images per class (quick runs)")
    a = p.parse_args()

    items = []
    for label, sub in ((0, "normal"), (1, "anomaly")):
        files = sorted(f for f in (a.data / sub).rglob("*") if f.suffix.lower() in EXTS)
        items += [(f, label) for f in files[: a.limit]]
    if not items:
        sys.exit(f"no images under {a.data}/normal or {a.data}/anomaly")

    rows = []
    for i, (f, label) in enumerate(items, 1):
        t0 = time.perf_counter()
        r = decide(a.server, a.head, a.state, a.question, OPTIONS, image=f)
        dt = time.perf_counter() - t0
        score = r["probabilities"]["Anomalous"]
        if not np.isfinite(score):
            sys.exit(f"NaN score on {f} - server state poisoned? restart with --no-mmproj-offload")
        rows.append({"path": str(f), "label": label, "score": score, "time_s": dt})
        print(f"[{i}/{len(items)}] {label} {score:.4f} {dt:.2f}s {f}", flush=True)

    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    y = [r["label"] for r in rows]
    s = [r["score"] for r in rows]
    print(f"\nN={len(rows)} (normal {y.count(0)}, anomaly {y.count(1)})")
    if 0 < sum(y) < len(y):
        print(f"AUROC: {auroc(y, s):.4f}")
    print("@0.5:    ", {k: round(v, 4) for k, v in binary_metrics(y, s, 0.5).items()})
    print("best F1: ", {k: round(v, 4) for k, v in best_f1(y, s).items()})
    print(f"mean time/img: {np.mean([r['time_s'] for r in rows]):.3f}s")
    print("\nTop-5 misjudged (|label - score| largest):")
    for r in sorted(rows, key=lambda r: -abs(r["label"] - r["score"]))[:5]:
        print(f"  label={r['label']} score={r['score']:.4f} {r['path']}")


if __name__ == "__main__":
    main()
