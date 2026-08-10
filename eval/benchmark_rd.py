#!/usr/bin/env python3
# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
Rate-Distortion Benchmark
=========================
Compares NeuroCodec against H.264, H.265, and AV1 across a range of bitrates.
Produces:
  - rd_results.json   (raw numbers)
  - rd_curve.pdf      (matplotlib figure)

Prerequisites: ffmpeg with libx265 and libaom-av1 support.

Usage:
    python eval/benchmark_rd.py --video dashcam.mp4 --output eval/results/
    python eval/benchmark_rd.py --video dashcam.mp4 --thresholds 0.05,0.1,0.2,0.3,0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Bitrate targets in kbps
DEFAULT_BITRATES = [100, 200, 400, 800, 1500, 3000]
# NeuroCodec threshold → approximate bitrate mapping (swept empirically)
DEFAULT_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Video quality metrics
# ---------------------------------------------------------------------------

def load_video_frames(path: str | Path, max_frames: int = 150) -> np.ndarray:
    cap    = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames) if frames else np.array([])


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    min_dim = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    ws = min(7, min_dim - (min_dim % 2 == 0))  # ensure odd window
    return float(ssim(img1, img2, data_range=1.0, win_size=max(3, ws)))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def avg_metrics(orig_frames: np.ndarray, recon_frames: np.ndarray) -> Dict[str, float]:
    n = min(len(orig_frames), len(recon_frames))
    if n == 0:
        return {"ssim": 0.0, "psnr": 0.0}
    ssims, psnrs = [], []
    for i in range(n):
        f1 = orig_frames[i]
        f2 = recon_frames[i]
        # Resize if needed
        if f1.shape != f2.shape:
            f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
        ssims.append(compute_ssim(f1, f2))
        psnrs.append(compute_psnr(f1, f2))
    return {"ssim": float(np.mean(ssims)), "psnr": float(np.mean(psnrs))}


def get_file_bitrate_kbps(path: str | Path, duration_s: float) -> float:
    size_bits = Path(path).stat().st_size * 8
    return size_bits / max(duration_s, 1.0) / 1000.0


def get_video_duration(path: str | Path) -> float:
    cap      = cv2.VideoCapture(str(path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n_frames / fps


# ---------------------------------------------------------------------------
# Standard codec baselines (via ffmpeg)
# ---------------------------------------------------------------------------

def encode_h264(input_path: Path, output_path: Path, bitrate_kbps: int) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264",
        "-b:v", f"{bitrate_kbps}k",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def encode_h265(input_path: Path, output_path: Path, bitrate_kbps: int) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx265",
        "-b:v", f"{bitrate_kbps}k",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def encode_av1(input_path: Path, output_path: Path, bitrate_kbps: int) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libaom-av1",
        "-b:v", f"{bitrate_kbps}k",
        "-cpu-used", "8",   # fastest preset for benchmarking
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# NeuroCodec benchmark
# ---------------------------------------------------------------------------

def benchmark_neurocodec(
    video_path: Path,
    orig_frames: np.ndarray,
    thresholds: List[float],
    weights_path: Optional[Path] = None,
) -> List[Dict]:
    from neurocodec.encoder import VoxelEncoder, NECReader
    from neurocodec.decoder import load_decoder
    import torch

    duration = get_video_duration(video_path)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = load_decoder(weights_path, device=device)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in thresholds:
            nec_path = Path(tmp) / f"out_c{c:.2f}.nec"
            enc      = VoxelEncoder(threshold=c)
            stats    = enc.encode(video_path, nec_path)
            bitrate  = get_file_bitrate_kbps(nec_path, duration)

            # Decode
            reader = NECReader(nec_path)
            dense  = reader.to_dense()  # (T, B, H, W)
            recon_frames = []
            model.reset_state()
            with torch.no_grad():
                for t in range(dense.shape[0]):
                    v    = torch.from_numpy(dense[t]).unsqueeze(0).to(device)
                    pred = model(v).squeeze().cpu().numpy()
                    recon_frames.append(pred)

            recon_arr = np.array(recon_frames)
            metrics   = avg_metrics(orig_frames, recon_arr)
            results.append({
                "codec":      "NeuroCodec",
                "threshold":  c,
                "bitrate_kbps": bitrate,
                **metrics,
            })
            logger.info("NeuroCodec C=%.2f | %.0f kbps | SSIM=%.4f | PSNR=%.2f dB",
                        c, bitrate, metrics["ssim"], metrics["psnr"])
    return results


# ---------------------------------------------------------------------------
# Standard codecs benchmark
# ---------------------------------------------------------------------------

def benchmark_standard_codecs(
    video_path: Path,
    orig_frames: np.ndarray,
    bitrates: List[int],
) -> List[Dict]:
    results = []
    codecs  = []

    if shutil.which("ffmpeg"):
        codecs.append(("H.264", encode_h264, "mp4"))
        codecs.append(("H.265", encode_h265, "mp4"))
        codecs.append(("AV1",   encode_av1,  "mp4"))
    else:
        logger.warning("ffmpeg not found — skipping standard codec baselines.")
        return results

    with tempfile.TemporaryDirectory() as tmp:
        for name, enc_fn, ext in codecs:
            for br in bitrates:
                out = Path(tmp) / f"{name.lower()}_{br}k.{ext}"
                ok  = enc_fn(video_path, out, br)
                if not ok:
                    logger.warning("Failed to encode %s at %d kbps", name, br)
                    continue
                recon = load_video_frames(out, max_frames=len(orig_frames))
                if len(recon) == 0:
                    continue
                metrics = avg_metrics(orig_frames, recon)
                actual_br = get_file_bitrate_kbps(out, get_video_duration(video_path))
                results.append({
                    "codec":        name,
                    "target_kbps":  br,
                    "bitrate_kbps": actual_br,
                    **metrics,
                })
                logger.info("%s @ %d kbps | SSIM=%.4f | PSNR=%.2f dB",
                            name, br, metrics["ssim"], metrics["psnr"])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rd_curve(results: List[Dict], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Rate-Distortion Curves: NeuroCodec vs Baselines", fontsize=14)

    codecs = sorted({r["codec"] for r in results})
    colors = {"NeuroCodec": "#1A56DB", "H.264": "#DC2626", "H.265": "#D97706", "AV1": "#16A34A"}
    markers = {"NeuroCodec": "o", "H.264": "s", "H.265": "^", "AV1": "D"}

    for metric, ax, label in [
        ("ssim", axes[0], "SSIM"),
        ("psnr", axes[1], "PSNR (dB)"),
    ]:
        for codec in codecs:
            pts = [(r["bitrate_kbps"], r[metric])
                   for r in results if r["codec"] == codec]
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys,
                    color=colors.get(codec, "gray"),
                    marker=markers.get(codec, "o"),
                    label=codec, linewidth=2, markersize=6)
        ax.set_xlabel("Bitrate (kbps)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Bitrate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    logger.info("Saved R-D curve to %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="NeuroCodec R-D benchmark")
    p.add_argument("--video",      required=True, help="Input video path")
    p.add_argument("--output",     default="eval/results",
                   help="Output directory (default: eval/results)")
    p.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.5",
                   help="NeuroCodec thresholds (comma-separated)")
    p.add_argument("--bitrates",   default="100,200,400,800,1500,3000",
                   help="Standard codec bitrates in kbps (comma-separated)")
    p.add_argument("--weights",    default=None,
                   help="Path to decoder weights (.pt)")
    p.add_argument("--max-frames", type=int, default=150,
                   help="Max frames to load for metrics (default 150)")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    bitrates   = [int(b)   for b in args.bitrates.split(",")]

    logger.info("Loading original frames from %s...", video_path.name)
    orig_frames = load_video_frames(video_path, max_frames=args.max_frames)
    logger.info("Loaded %d frames", len(orig_frames))

    all_results = []

    logger.info("\n=== NeuroCodec ===")
    all_results += benchmark_neurocodec(
        video_path, orig_frames, thresholds,
        weights_path=Path(args.weights) if args.weights else None,
    )

    logger.info("\n=== Standard Codecs ===")
    all_results += benchmark_standard_codecs(video_path, orig_frames, bitrates)

    # Save JSON
    json_path = output_dir / "rd_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved results to %s", json_path)

    # Plot
    plot_rd_curve(all_results, output_dir / "rd_curve.pdf")

    # Summary table
    print("\n── Rate-Distortion Summary ────────────────────────────────")
    print(f"{'Codec':<15} {'Bitrate (kbps)':>16} {'SSIM':>8} {'PSNR (dB)':>10}")
    print("─" * 55)
    for r in sorted(all_results, key=lambda x: (x["codec"], x["bitrate_kbps"])):
        print(f"{r['codec']:<15} {r['bitrate_kbps']:>16.0f} {r['ssim']:>8.4f} {r['psnr']:>10.2f}")
    print()


if __name__ == "__main__":
    main()
