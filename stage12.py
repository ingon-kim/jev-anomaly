"""Stage 1 (input tricks, no training) + Stage 2 (small classifier on Jev hidden state) for screw/bottle.

Protocol: test split -> dev/holdout halves (dev: pick threshold / fit classifier, holdout: report).
Server must be up with --no-mmproj-offload.
"""
import base64
import csv
import json
import sys
import time
from functools import cache
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "jev-omni-q4"))
from jev_omni_gguf_decide import decide, post_json  # noqa: E402

from eval_anomaly import auroc, best_f1, binary_metrics  # noqa: E402

SERVER = "http://127.0.0.1:8080"
HEAD_PATH = ROOT / "jev-omni-q4" / "decision-head-f32.npz"
HEAD = dict(np.load(HEAD_PATH))
CACHE = ROOT / "cache"
OUT = ROOT / "results" / "stage12"
PATCH, MAX_SIDE = 48, 33 * 48  # 33x33 = 1089 tokens <= 1120 limit

BASE_Q = "Is the object in the image normal, or does it have a defect or anomaly?"
P = {
    "bottle": dict(
        base=("The image shows a bottle photographed for industrial visual inspection.", BASE_Q),
        detail=("The image shows a glass bottle photographed from above for industrial visual inspection. "
                "Defects include a broken or chipped rim (large or small) and contamination or foreign material inside the bottle.",
                "Does this bottle have any break, chip, or contamination?"),
        multi=(["Normal", "Large break on the rim", "Small chip on the rim", "Contamination inside the bottle"],
               {"broken_large": 1, "broken_small": 2, "contamination": 3}),
    ),
    "screw": dict(
        base=("The image shows a screw photographed for industrial visual inspection.", BASE_Q),
        detail=("The image shows a metal screw photographed from above for industrial visual inspection. "
                "Defects include scratches on the head, damaged or deformed threads, and a manipulated tip.",
                "Does this screw have any scratch, thread damage, or deformation?"),
        multi=(["Normal", "Scratch on the head", "Scratch on the neck", "Damaged thread", "Manipulated or bent front tip"],
               {"scratch_head": 1, "scratch_neck": 2, "thread_side": 3, "thread_top": 3, "manipulated_front": 4}),
    ),
}
MULTI_Q = "Which option best describes the object in the image?"


@cache
def marker() -> str:
    return json.loads(urlopen(SERVER + "/props").read())["media_marker"]


def jev(img: Path, state: str, question: str, options: list[str]):
    """Same request as the adapter's decide(); returns (probabilities, last-token hidden state)."""
    choices = "\n".join(f"{i + 1}. {v}" for i, v in enumerate(options))
    text = (f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
            f"Reply with only the number of the correct option (1-{len(options)}).\nOutput a single number and nothing else.")
    prompt = f"<|turn>user\n{marker()}{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    r = post_json(SERVER + "/embedding", {"content": {"prompt_string": prompt, "multimodal_data": [base64.b64encode(img.read_bytes()).decode()]},
                                          "embd_normalize": -1})
    h = np.asarray(r[0]["embedding"][-1], dtype=np.float32)
    z = HEAD["linear.weight"][: len(options)] @ ((h - HEAD["mu"][0]) / HEAD["sd"][0]) + HEAD["linear.bias"][: len(options)]
    p = np.exp(z - z.max())
    p /= p.sum()
    if not np.isfinite(p).all():
        sys.exit(f"NaN on {img} - restart server with --no-mmproj-offload")
    return p, h


def roi(img: Path) -> Image.Image:
    """Crop to the object: pixels far from the border median colour, padded 8%."""
    im = Image.open(img).convert("RGB")
    a = np.asarray(im.convert("L"), dtype=float)
    border = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]))
    ys, xs = np.where(np.abs(a - border) > 40)
    if len(xs) < 100:
        return im
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # keep the square window inside the image: black padding differed between normal/anomaly and leaked a shortcut
    side = min(int(max(x1 - x0, y1 - y0) * 1.08), im.width, im.height)
    left = min(max((x0 + x1) // 2 - side // 2, 0), im.width - side)
    top = min(max((y0 + y1) // 2 - side // 2, 0), im.height - side)
    return im.crop((left, top, left + side, top + side))


def cached(img: Path, tag: str, make) -> Path:
    p = CACHE / tag / f"{img.parent.name}_{img.name}"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        make(img).save(p)
    return p


def upscaled(img):  # ponytail: upscaling adds no information, it only spreads the object over more 48px tokens
    return cached(img, "roi_up", lambda f: roi(f).resize((MAX_SIDE, MAX_SIDE), Image.BICUBIC))


def tiles(img):
    """2x2 overlapping tiles of the ROI (each 60% of side), each sent at 1008px."""
    out = []
    for k, (fx, fy) in enumerate([(0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)]):
        def make(f, fx=fx, fy=fy):
            im = roi(f)
            s = im.width
            return im.crop((int(fx * s), int(fy * s), int((fx + .6) * s), int((fy + .6) * s))).resize((1008, 1008), Image.BICUBIC)
        out.append(cached(img, f"tile{k}", make))
    return out


def split(cat):
    """Interleaved dev/holdout split by sorted filename (identical to the data/screw_dev|hold folders used in 5.1)."""
    rows = []
    for label, sub in ((0, "normal"), (1, "anomaly")):
        files = sorted((ROOT / "data" / cat / sub).glob("*.png"))
        for i, f in enumerate(files):
            rows.append(dict(path=f, label=label, defect=f.stem.rsplit("_", 1)[0], split="dev" if i % 2 == 0 else "hold"))
    return rows


def run_config(name, cat, rows, fn):
    """fn(row) -> (score, extra dict). Saves CSV; returns list of result dicts."""
    out_csv = OUT / f"{name}_{cat}.csv"
    res = []
    for i, r in enumerate(rows, 1):
        t0 = time.perf_counter()
        score, extra = fn(r)
        res.append(dict(path=str(r["path"]), label=r["label"], defect=r["defect"], split=r["split"],
                        score=float(score), time_s=time.perf_counter() - t0, **extra))
        if i % 40 == 0:
            print(f"  {name}/{cat} {i}/{len(rows)}", flush=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=res[0].keys())
        w.writeheader()
        w.writerows(res)
    return res


def summarize(name, cat, res):
    d = [r for r in res if r["split"] == "dev"]
    h = [r for r in res if r["split"] == "hold"]
    thr = best_f1([r["label"] for r in d], [r["score"] for r in d])["threshold"]
    m = binary_metrics([r["label"] for r in h], [r["score"] for r in h], thr)
    s = dict(config=name, cat=cat,
             dev_auroc=auroc([r["label"] for r in d], [r["score"] for r in d]),
             hold_auroc=auroc([r["label"] for r in h], [r["score"] for r in h]),
             dev_thr=thr, hold_f1=m["f1"], hold_prec=m["precision"], hold_rec=m["recall"],
             sec_per_img=float(np.mean([r["time_s"] for r in res])))
    if "type_ok" in res[0]:
        an = [r for r in h if r["label"] == 1]
        s["hold_type_acc"] = float(np.mean([r["type_ok"] for r in an]))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}, ensure_ascii=False), flush=True)
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # sanity: our request == adapter decide()
    probe = split("bottle")[0]["path"]
    st, q = P["bottle"]["base"]
    ref = decide(SERVER, HEAD_PATH, st, q, ["Normal", "Anomalous"], image=probe)["probabilities"]["Anomalous"]
    mine = jev(probe, st, q, ["Normal", "Anomalous"])[0][1]
    assert abs(ref - mine) < 1e-6, (ref, mine)
    print(f"sanity ok: adapter {ref:.6f} == ours {mine:.6f}", flush=True)

    summary = []
    only = set(sys.argv[1:])  # optional: run only these configs
    for cat in ("screw", "bottle"):
        rows = split(cat)
        cfg = P[cat]
        feats = {}

        def two(img, prompt, order=("Normal", "Anomalous")):
            p, h = jev(img, *prompt, list(order))
            return p[order.index("Anomalous")], h

        def c_base(r):
            s, h = two(r["path"], cfg["base"]); feats.setdefault("base", {})[str(r["path"])] = h
            return s, {}

        def c_detail(r):
            s, h = two(r["path"], cfg["detail"]); feats.setdefault("detail", {})[str(r["path"])] = h
            return s, {}

        def c_up(r):
            return two(upscaled(r["path"]), cfg["detail"])[0], {}

        def c_tiles(r):
            ts = [two(t, cfg["detail"])[0] for t in tiles(r["path"])]
            return max(ts), {"tile_scores": json.dumps([round(float(x), 4) for x in ts])}

        def c_swap(r):
            a = two(r["path"], cfg["detail"])[0]
            b = two(r["path"], cfg["detail"], ("Anomalous", "Normal"))[0]
            return (a + b) / 2, {"p_order1": float(a), "p_order2": float(b)}

        def c_multi(r):
            opts, dmap = cfg["multi"]
            p, _ = jev(r["path"], cfg["detail"][0], MULTI_Q, opts)
            pred = int(p.argmax())
            ok = int(pred == dmap.get(r["defect"], 0)) if r["label"] else int(pred == 0)
            return 1 - p[0], {"pred_type": opts[pred], "type_ok": ok}

        configs = [("S0_base", c_base), ("P1_detail", c_detail), ("U_roi_upscale", c_up),
                   ("T_tiles2x2", c_tiles), ("S_order_swap", c_swap), ("M_multi_option", c_multi)]
        for name, fn in configs:
            if only and name not in only:
                continue
            summary.append(summarize(name, cat, run_config(name, cat, rows, fn)))
        for k, v in feats.items():
            np.savez(OUT / f"hidden_{k}_{cat}.npz", paths=np.array(list(v)), H=np.stack(list(v.values())))

        # stage 2 needs train/good normals' hidden states (detail prompt) for the normal-only kNN
        tr = OUT / f"hidden_detail_train_{cat}.npz"
        if not tr.exists() and (not only or "stage2" in only):
            files = sorted((ROOT / "mvtec_raw" / "images" / "train" / cat / "good").glob("*.png"))
            H = []
            for i, f in enumerate(files, 1):
                H.append(jev(f, *cfg["detail"], ["Normal", "Anomalous"])[1])
                if i % 80 == 0:
                    print(f"  train features {cat} {i}/{len(files)}", flush=True)
            np.savez(tr, paths=np.array([str(f) for f in files]), H=np.stack(H))

    (OUT / "summary_stage1.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
