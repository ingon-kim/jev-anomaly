"""Gradio UI for Jev-Omni image anomaly check.

Needs llama-server on 127.0.0.1:8080 (./run.sh starts both). The server itself stays on loopback;
only this UI is exposed with --lan / --share.
"""
import argparse
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import gradio as gr
from PIL import Image

from stage12 import MAX_SIDE, P, SERVER, jev, roi

# thresholds = F1-optimal on the dev half (see docs/REPORT.md 6절). Custom input has no validated threshold.
PRESETS = {
    "bottle": dict(detail=P["bottle"]["detail"], base=P["bottle"]["base"], thr={"precise": 0.907, "fast": 0.625}),
    "screw": dict(detail=P["screw"]["detail"], base=P["screw"]["base"], thr={"precise": 0.987, "fast": 0.213}),
    "carpet": dict(detail=("The image shows a carpet photographed for industrial visual inspection.",
                           "Is the object in the image normal, or does it have a defect or anomaly?"),
                   base=("The image shows a carpet photographed for industrial visual inspection.",
                         "Is the object in the image normal, or does it have a defect or anomaly?"),
                   thr={"precise": 0.5, "fast": 0.5}),  # ponytail: carpet precise-mode threshold not validated
}
MODES = {"정밀 (물체 확대, ~0.8s)": "precise", "빠름 (원본, ~0.4s)": "fast"}


def fill(preset, mode):
    if preset == "직접 입력":
        return gr.update(), gr.update(), gr.update()
    p = PRESETS[preset]
    m = MODES[mode]
    st, q = p["detail"] if m == "precise" else p["base"]
    return st, q, p["thr"][m]


def inspect(image_path, mode, state, question, thr):
    if not image_path:
        raise gr.Error("이미지를 올려 주세요.")
    t0 = time.perf_counter()
    if MODES[mode] == "precise":
        sent = roi(image_path).resize((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
    else:
        sent = Image.open(image_path).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        sent.save(f.name)
        try:
            p, _ = jev(Path(f.name), state, question, ["Normal", "Anomalous"])
        except SystemExit as e:  # NaN guard in jev()
            raise gr.Error(str(e))
    score, dt = float(p[1]), time.perf_counter() - t0
    bad = score >= thr
    verdict = f"## {'🔴 이상 (Anomalous)' if bad else '🟢 정상 (Normal)'}\n" \
              f"P(Anomalous) = **{score:.4f}** · 기준선 {thr:.3f} · {dt:.2f}s"
    return verdict, {"Anomalous": score, "Normal": 1 - score}, sent


def server_status():
    try:
        urlopen(SERVER + "/health", timeout=3)
        return "✅ llama-server 연결됨"
    except OSError:
        return "❌ llama-server에 연결할 수 없음 — `./run.sh`로 서버부터 띄우세요"


with gr.Blocks(title="Jev-Omni 이상 탐지") as demo:
    gr.Markdown("# Jev-Omni 이미지 이상 탐지\n사진을 올리면 Jev-Omni가 *정상 / 이상* 확률을 매깁니다. 모델은 학습 없이 질문 문장으로만 동작합니다.")
    status = gr.Markdown()
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="검사할 이미지")
            preset = gr.Radio(["bottle", "screw", "carpet", "직접 입력"], value="screw", label="품목 프리셋")
            mode = gr.Radio(list(MODES), value=list(MODES)[0], label="모드")
            state = gr.Textbox(label="상황 설명 (state)", lines=3)
            question = gr.Textbox(label="질문 (question)")
            thr = gr.Slider(0, 1, step=0.001, label="기준선: P(Anomalous) ≥ 이 값이면 이상")
            run = gr.Button("검사", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            probs = gr.Label(label="확률", show_heading=False)  # heading = argmax, conflicts with the threshold verdict
            sent = gr.Image(label="모델에 실제로 들어간 이미지", interactive=False)
            gr.Markdown("기준선은 MVTec AD dev 절반에서 정한 값입니다. 실제 라인 이미지에는 반드시 다시 보정하세요. "
                        "`직접 입력`은 검증된 기준선이 없습니다.")
    # sample images from ./get_mvtec.sh (MVTec AD, CC BY-NC-SA - not shipped in the repo); hidden if not downloaded
    samples = [(f"data/{c}/{sub}/{f}", c, label) for c, sub, f, label in [
        ("screw", "anomaly", "scratch_head_004.png", "screw · 불량: 머리 긁힘"),
        ("screw", "anomaly", "thread_top_000.png", "screw · 불량: 나사산 손상"),
        ("screw", "normal", "good_010.png", "screw · 정상"),
        ("bottle", "anomaly", "broken_large_004.png", "bottle · 불량: 크게 깨짐"),
        ("bottle", "anomaly", "contamination_016.png", "bottle · 불량: 내부 오염"),
        ("bottle", "normal", "good_014.png", "bottle · 정상"),
        ("carpet", "anomaly", "hole_007.png", "carpet · 불량: 구멍"),
        ("carpet", "normal", "good_020.png", "carpet · 정상"),
    ] if Path(f"data/{c}/{sub}/{f}").exists()]
    if samples:
        gr.Examples([[p, c, list(MODES)[0]] for p, c, _ in samples], [image, preset, mode],
                    [state, question, thr], fn=lambda _img, p, m: fill(p, m), run_on_click=True,
                    example_labels=[lab for *_, lab in samples], label="테스트 샘플 (MVTec AD) — 클릭 후 [검사]")
    for c in (preset, mode):
        c.change(fill, [preset, mode], [state, question, thr])
    demo.load(fill, [preset, mode], [state, question, thr]).then(server_status, None, status)
    run.click(inspect, [image, mode, state, question, thr], [verdict, probs, sent])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan", action="store_true", help="listen on 0.0.0.0 (same network access)")
    ap.add_argument("--share", action="store_true", help="temporary public gradio.live link (requires --auth)")
    ap.add_argument("--auth", help="user:password")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if a.share and not a.auth:
        ap.error("--share exposes the app publicly; set --auth user:password")
    auth = tuple(a.auth.split(":", 1)) if a.auth else None
    demo.queue(default_concurrency_limit=1)  # one llama-server slot (--parallel 1)
    demo.launch(server_name="0.0.0.0" if a.lan else "127.0.0.1", server_port=a.port, share=a.share, auth=auth)
