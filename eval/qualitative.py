#!/usr/bin/env python3
# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
Qualitative Comparison Script
==============================
Generates side-by-side frames: Original | H.264 | NeuroCodec
for use as Figure 3 in the paper.

Usage:
    python eval/qualitative.py --video dashcam.mp4 --output eval/results/qual/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def make_comparison_grid(
    original: np.ndarray,
    h264: np.ndarray,
    neurocodec: np.ndarray,
    labels: list[str] = None,
) -> np.ndarray:
    """Stack three frames side-by-side with labels."""
    if labels is None:
        labels = ["Original", "H.264 @200kbps", "NeuroCodec @200kbps"]

    frames = [original, h264, neurocodec]
    H, W = frames[0].shape[:2]

    # Resize all to same size
    frames = [cv2.resize(f, (W, H)) for f in frames]

    # Convert grayscale to BGR for uniform stacking
    frames_bgr = []
    for f in frames:
        if f.ndim == 2:
            frames_bgr.append(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        else:
            frames_bgr.append(f)

    # Add label overlays
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (frame, label) in enumerate(zip(frames_bgr, labels)):
        cv2.rectangle(frame, (0, H - 28), (W, H), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, H - 8), font, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Separator lines
    sep = np.zeros((H, 3, 3), dtype=np.uint8)
    sep[:] = [80, 80, 80]

    grid = np.hstack([frames_bgr[0], sep, frames_bgr[1], sep, frames_bgr[2]])
    return grid


def main() -> None:
    p = argparse.ArgumentParser(description="Qualitative codec comparison")
    p.add_argument("--video",   required=True, help="Input video")
    p.add_argument("--output",  default="eval/results/qual",
                   help="Output directory")
    p.add_argument("--bitrate", type=int, default=200,
                   help="Comparison bitrate in kbps (default 200)")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="NeuroCodec threshold (default 0.2)")
    p.add_argument("--weights", default=None, help="Decoder weights (.pt)")
    p.add_argument("--allow-random", action="store_true",
                   help="Use an untrained decoder (output is meaningless)")
    p.add_argument("--n-frames", type=int, default=5,
                   help="Number of comparison frames to extract")
    args = p.parse_args()

    video_path = Path(args.video)
    out_dir    = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # ── Encode H.264 baseline ────────────────────────────────────────
        h264_path = Path(tmp) / "h264.mp4"
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg not found on PATH (needed for the H.264 baseline)")
        res = subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-b:v", f"{args.bitrate}k",
            "-an", str(h264_path),
        ], capture_output=True)
        if res.returncode != 0:
            raise SystemExit("ffmpeg H.264 encode failed:\n" + res.stderr.decode(errors="replace"))

        # ── Encode NeuroCodec ────────────────────────────────────────────
        from neurocodec.encoder import VoxelEncoder, NECReader
        from neurocodec.decoder import load_decoder
        import torch

        nec_path = Path(tmp) / "encoded.nec"
        enc      = VoxelEncoder(threshold=args.threshold)
        enc.encode(video_path, nec_path)

        reader = NECReader(nec_path)
        device = torch.device("cpu")
        model  = load_decoder(args.weights, device=device, n_bins=reader.n_bins,
                              require_weights=not args.allow_random)

        recon_path = Path(tmp) / "neurocodec_recon.mp4"
        H, W = reader.height, reader.width
        writer_nc = cv2.VideoWriter(
            str(recon_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H), False)
        model.reset_state()
        with torch.inference_mode():
            for voxel in reader.iter_dense():
                v   = torch.from_numpy(voxel).unsqueeze(0)
                out = model(v)[0, 0].clamp(0, 1).cpu().numpy()
                writer_nc.write((out * 255).astype(np.uint8))
        writer_nc.release()

        # ── Extract frames and build comparison grids ────────────────────
        cap_orig = cv2.VideoCapture(str(video_path))
        cap_h264 = cv2.VideoCapture(str(h264_path))
        cap_nc   = cv2.VideoCapture(str(recon_path))

        n_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        step     = max(1, n_frames // args.n_frames)

        saved = 0
        for frame_idx in range(0, n_frames, step):
            if saved >= args.n_frames:
                break
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_h264.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_nc.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret1, f_orig = cap_orig.read()
            ret2, f_h264 = cap_h264.read()
            ret3, f_nc   = cap_nc.read()
            if not (ret1 and ret2 and ret3):
                continue

            f_nc_bgr = cv2.cvtColor(f_nc, cv2.COLOR_GRAY2BGR) \
                       if f_nc.ndim == 2 else f_nc

            grid = make_comparison_grid(
                f_orig, f_h264, f_nc_bgr,
                labels=["Original",
                        f"H.264 @{args.bitrate}kbps",
                        f"NeuroCodec C={args.threshold}"])

            out_path = out_dir / f"comparison_frame{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved {out_path}")
            saved += 1

        cap_orig.release()
        cap_h264.release()
        cap_nc.release()

    print(f"\n{saved} comparison frames written to {out_dir}/")


if __name__ == "__main__":
    main()
