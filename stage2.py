"""Stage 2: small classifiers on Jev-Omni last-token hidden states (backbone untouched).

Reads hidden states saved by stage12.py. Same dev/holdout protocol.
- LR:  logistic regression fit on dev (labels), dev scores via 5-fold CV so the dev threshold isn't in-sample
- LR+: same, plus MVTec train/good normals as extra label-0 samples
- kNN: normal-only; score = mean cosine distance to k nearest train/good normals (k picked on dev)
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from eval_anomaly import auroc
from stage12 import HEAD, OUT, split, summarize


def load(name):
    z = np.load(OUT / name)
    return {p: h for p, h in zip(z["paths"], z["H"])}


def std(H):  # the decision head's own standardisation
    return (H - HEAD["mu"][0]) / HEAD["sd"][0]


def main():
    summary = []
    for cat in ("screw", "bottle"):
        rows = split(cat)
        feats = load(f"hidden_detail_{cat}.npz")
        X = std(np.stack([feats[str(r["path"])] for r in rows]))
        y = np.array([r["label"] for r in rows])
        dev = np.array([r["split"] == "dev" for r in rows])
        tr = std(np.load(OUT / f"hidden_detail_train_{cat}.npz")["H"])

        def save(name, scores):
            res = [dict(path=str(r["path"]), label=r["label"], defect=r["defect"], split=r["split"], score=float(s), time_s=0.0)
                   for r, s in zip(rows, scores)]
            with (OUT / f"{name}_{cat}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=res[0].keys())
                w.writeheader()
                w.writerows(res)
            s = summarize(name, cat, res)
            s.pop("sec_per_img")
            summary.append(s)

        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        for name, Xtr, ytr in (("L_logreg_dev", X[dev], y[dev]),
                               ("L_logreg_dev+trainnormal", np.vstack([X[dev], tr]), np.r_[y[dev], np.zeros(len(tr), int)])):
            clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000)
            scores = np.empty(len(rows))
            # dev: out-of-fold predictions; extra train normals always stay in the training folds
            n_dev = dev.sum()
            oof = np.empty(n_dev)
            for a, b in cv.split(X[dev], y[dev]):
                fit_idx = np.r_[a, np.arange(n_dev, len(Xtr))]
                oof[b] = clf.fit(Xtr[fit_idx], ytr[fit_idx]).predict_proba(X[dev][b])[:, 1]
            scores[dev] = oof
            scores[~dev] = clf.fit(Xtr, ytr).predict_proba(X[~dev])[:, 1]
            save(name, scores)

        # normal-only kNN on cosine distance
        def unit(A):
            return A / np.linalg.norm(A, axis=1, keepdims=True)
        D = 1 - unit(X) @ unit(tr).T
        best_k = max((1, 3, 5, 10), key=lambda k: auroc(y[dev], np.sort(D[dev], 1)[:, :k].mean(1)))
        save(f"K_knn_trainnormal_k{best_k}", np.sort(D, 1)[:, :best_k].mean(1))

    (OUT / "summary_stage2.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
