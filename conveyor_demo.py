"""Synthetic conveyor demo: MVTec test images ride a belt, Jev-Omni judges each at the camera.

Real model call per item (server must be up with --no-mmproj-offload); video is rendered offline.
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "jev-omni-q4"))
from jev_omni_gguf_decide import decide

HEAD = Path(__file__).parent / "jev-omni-q4" / "decision-head-f32.npz"
Q = "Is the object in the image normal, or does it have a defect or anomaly?"
# prompt + threshold per product, from docs/REPORT.md (thresholds picked on test/dev data -> optimistic)
PRESETS = {
    "bottle": ("The image shows a bottle photographed for industrial visual inspection.", Q, 0.625),
    "carpet": ("The image shows a carpet photographed for industrial visual inspection.", Q, 0.5),
    "screw": ("The image shows a metal screw photographed from above for industrial visual inspection. "
              "Defects include scratches on the head, damaged or deformed threads, and a manipulated tip.",
              "Does this screw have any scratch, thread damage, or deformation?", 0.983),
}
W, H, FPS, SPEED = 1280, 560, 30, 4
ITEM, GAP = 220, 340
BELT_Y0, BELT_Y1 = 200, 470
CAM_X = W // 2
GREEN, RED, GRAY, WHITE = (40, 200, 90), (230, 50, 50), (150, 150, 150), (240, 240, 240)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(size):
    return ImageFont.truetype(FONT, size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cat", default="bottle", choices=PRESETS)
    p.add_argument("--n", type=int, default=12, help="items (half normal, half anomaly)")
    p.add_argument("--thr", type=float, help="override threshold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    state, question, thr = PRESETS[a.cat]
    thr = a.thr if a.thr is not None else thr
    out = a.out or Path(f"demo_{a.cat}.mp4")

    rng = random.Random(a.seed)
    items = []
    for label, sub in ((0, "normal"), (1, "anomaly")):
        files = sorted((Path("data") / a.cat / sub).glob("*.png"))
        items += [(f, label) for f in rng.sample(files, a.n // 2)]
    rng.shuffle(items)

    # 1) real inference per item
    judged = []
    for i, (f, label) in enumerate(items, 1):
        t0 = time.perf_counter()
        score = decide("http://127.0.0.1:8080", HEAD, state, question, ["Normal", "Anomalous"], image=f)["probabilities"]["Anomalous"]
        dt = time.perf_counter() - t0
        defect = f.stem.rsplit("_", 1)[0]
        img = Image.open(f).convert("RGB").resize((ITEM, ITEM))
        judged.append(dict(img=img, label=label, defect=defect, score=score, dt=dt, pred=int(score >= thr)))
        print(f"[{i}/{len(items)}] gt={label} {defect:22s} score={score:.3f} pred={int(score >= thr)} {dt:.2f}s", flush=True)

    # 2) render
    f_big, f_mid, f_small = font(34), font(24), font(19)
    total_px = W + 2 * ITEM + (len(judged) - 1) * GAP
    n_frames = total_px // SPEED + FPS * 2
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE)
    cam_t = {}  # item index -> frame when its center hit the camera
    for fr in range(n_frames):
        im = Image.new("RGB", (W, H), (28, 30, 34))
        d = ImageDraw.Draw(im)
        # belt with moving slats
        d.rectangle([0, BELT_Y0, W, BELT_Y1], fill=(55, 58, 64))
        for x in range(-(fr * SPEED) % 60 - 60, W, 60):
            d.line([x, BELT_Y0, x, BELT_Y1], fill=(70, 74, 80), width=3)

        done = []
        for i, it in enumerate(judged):
            cx = -ITEM // 2 + fr * SPEED - i * GAP
            if cx >= CAM_X and i not in cam_t:
                cam_t[i] = fr
            status = None
            if i in cam_t:
                elapsed = (fr - cam_t[i]) / FPS
                status = "busy" if elapsed < it["dt"] else "done"
                if status == "done":
                    done.append(it)
            if not -ITEM < cx < W + ITEM:
                continue
            x0, y0 = cx - ITEM // 2, (BELT_Y0 + BELT_Y1) // 2 - ITEM // 2
            im.paste(it["img"], (x0, y0))
            if status == "busy":
                d.rectangle([x0 - 4, y0 - 4, x0 + ITEM + 4, y0 + ITEM + 4], outline=(90, 160, 255), width=5)
                d.text((cx, y0 - 12), "검사 중…", font=f_mid, fill=(90, 160, 255), anchor="mb")
            elif status == "done":
                c = RED if it["pred"] else GREEN
                d.rectangle([x0 - 5, y0 - 5, x0 + ITEM + 5, y0 + ITEM + 5], outline=c, width=7)
                d.text((cx, y0 - 44), f"{'이상' if it['pred'] else '정상'}  {it['score']:.2f}", font=f_big, fill=c, anchor="mb")
                ok = it["pred"] == it["label"]
                gt = f"실제: {'이상 (' + it['defect'] + ')' if it['label'] else '정상'}  {'맞음' if ok else '오판'}"
                d.text((cx, y0 - 12), gt, font=f_small, fill=WHITE if ok else (255, 200, 60), anchor="mb")

        # camera zone (drawn over items)
        d.rectangle([CAM_X - ITEM // 2 - 14, BELT_Y0 - 10, CAM_X + ITEM // 2 + 14, BELT_Y1 + 10], outline=(90, 160, 255), width=3)
        d.text((CAM_X, BELT_Y1 + 18), "검사 카메라", font=f_small, fill=(90, 160, 255), anchor="mt")

        # HUD
        n_ok = sum(it["pred"] == it["label"] for it in done)
        d.text((24, 20), f"Jev-Omni 컨베이어 검사 데모 — {a.cat}", font=f_big, fill=WHITE)
        d.text((24, 70), f"임계값 P(이상) ≥ {thr:.3f}   |   검사 {len(done)}/{len(judged)}   |   정답 {n_ok}/{len(done) or 0}"
               f"   |   평균 추론 {sum(it['dt'] for it in judged) / len(judged):.2f}s/개", font=f_mid, fill=GRAY)
        d.text((24, 110), "초록=정상 판정, 빨강=이상 판정 · 점수=P(Anomalous) · 실제 모델 추론 결과", font=f_small, fill=GRAY)
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"saved {out}  ({n_frames / FPS:.1f}s)")


if __name__ == "__main__":
    main()
