# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
NeuroCodec — HuggingFace Spaces Demo
=====================================
Live demo: upload any video → see DVS events → see reconstructed frames
with SSIM score and compression ratio.

Deploy to HuggingFace Spaces (free, always-on):
    1. Create a new Space at huggingface.co/new-space
    2. SDK: Gradio
    3. Upload this file as app.py
    4. Upload neurocodec/ package folder (or pip install neurocodec)
    5. Optional: add decoder_v1.pt to the Space files

Run locally:
    pip install gradio neurocodec
    python demo/app.py
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurocodec.demo")

# ── Try to import neurocodec (installed or local) ─────────────────────────
try:
    from neurocodec import decode
    from neurocodec.simulator import DVSSimulator, save_event_video
    from neurocodec.encoder import VoxelEncoder
    NEUROCODEC_AVAILABLE = True
except ImportError:
    NEUROCODEC_AVAILABLE = False
    logger.warning("neurocodec not installed — pip install neurocodec")

# ── Example clips bundled with the Space ──────────────────────────────────
EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLES = []
if EXAMPLES_DIR.exists():
    for ext in ("*.mp4", "*.avi", "*.mov"):
        EXAMPLES.extend(sorted(EXAMPLES_DIR.glob(ext)))

# ─────────────────────────────────────────────────────────────────────────
# Core processing functions
# ─────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    threshold: float,
    n_bins: int,
    k_frac: float,
    max_frames: int,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Run the full NeuroCodec pipeline on an uploaded video.

    Returns
    -------
    event_video_path  : str | None   Path to event visualisation video.
    recon_video_path  : str | None   Path to reconstructed video.
    info_text         : str          Stats and metadata.
    """
    if not NEUROCODEC_AVAILABLE:
        return None, None, "❌ neurocodec not installed. Run: pip install neurocodec"

    if video_path is None:
        return None, None, "⬆️ Upload a video to get started."

    tmpdir = tempfile.mkdtemp(prefix="neurocodec_")
    try:
        video_path = Path(video_path)

        # ── 1. Truncate to max_frames for speed ───────────────────────
        truncated_path = Path(tmpdir) / "input_truncated.mp4"
        _truncate_video(video_path, truncated_path, max_frames)

        # ── 2. Simulate events ────────────────────────────────────────
        logger.info("Simulating events (C=%.2f)...", threshold)
        sim    = DVSSimulator(threshold=threshold)
        stream = sim.simulate_video(truncated_path)

        event_vid_path = Path(tmpdir) / "events.mp4"
        save_event_video(stream, event_vid_path)

        # ── 3. Encode to .nec ─────────────────────────────────────────
        nec_path = Path(tmpdir) / "encoded.nec"
        enc      = VoxelEncoder(threshold=threshold, n_bins=n_bins, k_frac=k_frac)
        stats    = enc.encode(truncated_path, nec_path)

        # ── 4. Decode back to video (needs trained weights) ───────────
        recon_path = Path(tmpdir) / "reconstructed.mp4"
        out_recon, ssim_score, note = None, None, ""
        try:
            decode(nec_path, recon_path)
            ssim_score = _compute_avg_ssim(truncated_path, recon_path, n=30)
            out_recon = _persist(recon_path)
        except FileNotFoundError as e:
            note = f"⚠️ Reconstruction skipped: {e}"

        # ── 5. Copy outputs out of the temp dir (Gradio serves them later) ──
        out_event = _persist(event_vid_path)

        info = _format_stats(stats, ssim_score, stream, threshold, n_bins, k_frac)
        if note:
            info = note + "\n\n" + info
        return out_event, out_recon, info

    except Exception as e:
        logger.exception("Processing failed")
        return None, None, f"❌ Error: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _persist(src: Path) -> str:
    """Copy ``src`` to a unique file in the system temp dir that outlives the request."""
    fd, dst = tempfile.mkstemp(prefix="neurocodec_out_", suffix=src.suffix)
    os.close(fd)
    shutil.copy(src, dst)
    return dst


def _truncate_video(src: Path, dst: Path, max_frames: int) -> None:
    cap  = cv2.VideoCapture(str(src))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1
    cap.release()
    out.release()


def _compute_avg_ssim(orig_path: Path, recon_path: Path, n: int = 30) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return -1.0

    cap1 = cv2.VideoCapture(str(orig_path))
    cap2 = cv2.VideoCapture(str(recon_path))
    scores = []
    for _ in range(n):
        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        if not r1 or not r2:
            break
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if g1.shape != g2.shape:
            g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
        scores.append(float(ssim(g1, g2, data_range=1.0)))
    cap1.release()
    cap2.release()
    return float(np.mean(scores)) if scores else 0.0


def _format_stats(stats, ssim, stream, threshold, n_bins, k_frac) -> str:
    lines = [
        "## NeuroCodec Results",
        "",
        f"**Threshold C:** {threshold:.2f}  |  **Bins B:** {n_bins}  |  **Top-K:** {k_frac:.0%}",
        "",
        "### Encoding",
        f"- Events simulated: **{stats['n_events']:,}**",
        f"- Event density: **{stats['event_density']:.4f}**",
        f"- Frames processed: **{stats['n_frames']}**",
        f"- Encoded size: **{stats['file_size_kb']:.1f} KB**",
        f"- Size vs input file: **{stats['compression_ratio']:.2f}×** (values below 1 mean the .nec is larger)",
        "",
        "### Quality",
        "- SSIM: *n/a (no trained decoder weights available)*" if ssim is None else
        (f"- SSIM (30-frame avg): **{ssim:.4f}**" if ssim >= 0
         else "- SSIM: *scikit-image not available*"),
        "",
        "### What you're seeing",
        "- **Left video:** DVS event visualisation — red dots = positive polarity (brighter), "
        "blue dots = negative (darker)",
        "- **Right video:** ConvLSTM reconstructed frames from the encoded .nec file",
        "",
        "> 💡 Lower threshold → more events → better quality, less compression.  ",
        "> Higher threshold → fewer events → more compression, lower quality.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────

CSS = """
#header { text-align: center; margin-bottom: 1rem; }
#header h1 { font-size: 2.2rem; font-weight: 700; }
#header p  { color: #666; }
.result-video { border-radius: 8px; }
"""

with gr.Blocks(css=CSS, title="NeuroCodec Demo") as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("""
# NeuroCodec 🧠⚡
**Neuromorphic-Inspired Spike-Driven Video Compression**

Upload any video → get DVS event visualisation + ConvLSTM reconstruction + compression stats.  
No event-camera hardware required — runs entirely on standard video.

[![PyPI](https://img.shields.io/pypi/v/neurocodec)](https://pypi.org/project/neurocodec/)
[![GitHub](https://img.shields.io/badge/GitHub-neurocodec-blue)](https://github.com/chandini-narayana/neurocodec)
""")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video", elem_classes="result-video")

            with gr.Accordion("⚙️ Encoding Parameters", open=False):
                threshold = gr.Slider(
                    minimum=0.05, maximum=1.0, value=0.2, step=0.05,
                    label="Threshold C — log-intensity change that fires an event",
                    info="Lower = more events = better quality. Higher = more compression."
                )
                n_bins = gr.Slider(
                    minimum=2, maximum=10, value=5, step=1,
                    label="Time Bins B — voxel grid temporal resolution",
                )
                k_frac = gr.Slider(
                    minimum=0.05, maximum=1.0, value=0.5, step=0.05,
                    label="Top-K fraction — share of non-zero voxel cells kept",
                    info="0.5 = keep the strongest 50% of active cells."
                )
                max_frames = gr.Slider(
                    minimum=10, maximum=150, value=60, step=10,
                    label="Max frames to process (for demo speed)",
                )

            run_btn = gr.Button("▶  Run NeuroCodec", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Row():
                event_vid  = gr.Video(label="DVS Event Visualisation", elem_classes="result-video")
                recon_vid  = gr.Video(label="Reconstructed Video",       elem_classes="result-video")
            stats_md = gr.Markdown("*Results will appear here after processing.*")

    # Examples
    if EXAMPLES:
        gr.Examples(
            examples=[[str(p)] for p in EXAMPLES],
            inputs=[video_input],
            label="Try an example clip",
        )

    gr.Markdown("""
---
### How it works
1. **DVS Simulator** — detects per-pixel log-intensity changes above threshold C, produces (x, y, t, polarity) events
2. **Voxel Grid** — accumulates the events of the last B frames into a [B × H × W] tensor per frame
3. **Top-K Sparsification** — keeps the strongest K fraction of non-zero voxel cells
4. **Entropy Coding** — delta-codes the indices, quantises values to int8 and DEFLATE-compresses each frame → .nec binary format
5. **ConvLSTM Decoder** — trained neural network reconstructs frames from voxel grids

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) |
**Code:** [github.com/chandini-narayana/neurocodec](https://github.com/chandini-narayana/neurocodec) |
**Author:** Chandini Narayana (2026)
""")

    run_btn.click(
        fn=process_video,
        inputs=[video_input, threshold, n_bins, k_frac, max_frames],
        outputs=[event_vid, recon_vid, stats_md],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
