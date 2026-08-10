# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
DVS Event Simulator
===================
Simulates Dynamic Vision Sensor (DVS) events from standard video by
detecting per-pixel log-intensity changes above a threshold C.

Each event is a 4-tuple (x, y, t, polarity) where polarity is +1 (brighter)
or -1 (darker).  This matches the DAVIS event-camera format so outputs are
compatible with E2VID, ESIM, and the DAVIS240C dataset.

Reference:
    Rebecq et al., "ESIM: an Open Event Camera Simulator", CoRL 2018.
    Gallego et al., "Event-based Vision: A Survey", TPAMI 2022.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Single DVS event."""
    x: int
    y: int
    t: int          # frame index (0-based)
    polarity: int   # +1 bright, -1 dark


@dataclass
class EventStream:
    """Container for a sequence of events produced by the simulator."""
    events: List[Event] = field(default_factory=list)
    width: int = 0
    height: int = 0
    n_frames: int = 0
    threshold: float = 0.2
    fps: float = 30.0

    def __len__(self) -> int:
        return len(self.events)

    def density(self) -> float:
        """Fraction of pixels that fired at least one event."""
        if self.n_frames == 0 or self.width == 0 or self.height == 0:
            return 0.0
        return len(self.events) / max(1, self.n_frames * self.width * self.height)

    def to_numpy(self) -> np.ndarray:
        """Return Nx4 array of (x, y, t, polarity)."""
        if not self.events:
            return np.zeros((0, 4), dtype=np.int32)
        return np.array(
            [(e.x, e.y, e.t, e.polarity) for e in self.events], dtype=np.int32
        )


# ---------------------------------------------------------------------------
# Core simulator class
# ---------------------------------------------------------------------------

class DVSSimulator:
    """
    Per-frame log-intensity DVS event simulator.

    Parameters
    ----------
    threshold : float
        Log-intensity change magnitude that triggers an event.
        Lower  → more events (finer detail, less compression).
        Higher → fewer events (more compression, more error).
        Typical range: 0.05 – 0.5.  Default 0.2 is a good starting point.
    epsilon : float
        Small value added before log to avoid log(0).  Default 1e-6.
    gray_method : str
        How to convert BGR frames to grayscale.
        'cv2'  – fast OpenCV default.
        'luma' – BT.601 weighted (Y = 0.299R + 0.587G + 0.114B), closer
                 to perceived brightness and human retina response.
    """

    def __init__(
        self,
        threshold: float = 0.2,
        epsilon: float = 1e-6,
        gray_method: str = "luma",
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.threshold = threshold
        self.epsilon = epsilon
        self.gray_method = gray_method

    # ------------------------------------------------------------------
    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 frame to float32 in [0, 1]."""
        if self.gray_method == "luma":
            r = frame_bgr[:, :, 2].astype(np.float32)
            g = frame_bgr[:, :, 1].astype(np.float32)
            b = frame_bgr[:, :, 0].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray / 255.0

    def _log_intensity(self, gray: np.ndarray) -> np.ndarray:
        return np.log(gray + self.epsilon)

    # ------------------------------------------------------------------
    def simulate_array(
        self, frames: np.ndarray, fps: float = 30.0
    ) -> EventStream:
        """
        Simulate events from a pre-loaded array of BGR frames.

        Parameters
        ----------
        frames : np.ndarray  shape (T, H, W, 3), dtype uint8
        fps    : float

        Returns
        -------
        EventStream
        """
        T, H, W, _ = frames.shape
        stream = EventStream(width=W, height=H, n_frames=T,
                             threshold=self.threshold, fps=fps)

        prev_log = self._log_intensity(self._to_gray(frames[0]))

        for t in range(1, T):
            curr_gray = self._to_gray(frames[t])
            curr_log = self._log_intensity(curr_gray)
            diff = curr_log - prev_log

            pos_ys, pos_xs = np.where(diff > self.threshold)
            neg_ys, neg_xs = np.where(diff < -self.threshold)

            for x, y in zip(pos_xs, pos_ys):
                stream.events.append(Event(int(x), int(y), t, +1))
            for x, y in zip(neg_xs, neg_ys):
                stream.events.append(Event(int(x), int(y), t, -1))

            prev_log = curr_log

        logger.info(
            "Simulated %d events from %d frames (%dx%d) | density=%.4f",
            len(stream), T, W, H, stream.density(),
        )
        return stream

    # ------------------------------------------------------------------
    def simulate_video(self, video_path: str | Path) -> EventStream:
        """
        Simulate events frame-by-frame from a video file.

        Streams frames one at a time — does not load the whole video into RAM.
        Suitable for long clips on consumer hardware.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stream = EventStream(width=width, height=height,
                             threshold=self.threshold, fps=fps)

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Could not read first frame.")

        prev_log = self._log_intensity(self._to_gray(frame))
        t = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            curr_gray = self._to_gray(frame)
            curr_log  = self._log_intensity(curr_gray)
            diff      = curr_log - prev_log

            pos_ys, pos_xs = np.where(diff > self.threshold)
            neg_ys, neg_xs = np.where(diff < -self.threshold)

            for x, y in zip(pos_xs, pos_ys):
                stream.events.append(Event(int(x), int(y), t, +1))
            for x, y in zip(neg_xs, neg_ys):
                stream.events.append(Event(int(x), int(y), t, -1))

            prev_log = curr_log
            t += 1

        cap.release()
        stream.n_frames = t
        logger.info(
            "Simulated %d events from %s (%dx%d, %d frames, C=%.2f)",
            len(stream), path.name, width, height, t, self.threshold,
        )
        return stream

    # ------------------------------------------------------------------
    def threshold_sweep(
        self, video_path: str | Path, thresholds: Optional[list] = None
    ) -> dict:
        """
        Sweep over multiple threshold values and return event counts.
        Useful for plotting event-density vs C (your first paper figure).
        """
        if thresholds is None:
            thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
        results = {}
        for c in thresholds:
            sim = DVSSimulator(threshold=c)
            stream = sim.simulate_video(video_path)
            results[c] = {
                "n_events":  len(stream),
                "density":   stream.density(),
                "n_frames":  stream.n_frames,
            }
        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def simulate(
    video_path: str | Path,
    threshold: float = 0.2,
    gray_method: str = "luma",
) -> EventStream:
    """
    Simulate DVS events from a video file.

    Parameters
    ----------
    video_path : str | Path
    threshold  : float   Log-intensity change threshold (default 0.2).
    gray_method: str     'luma' or 'cv2'.

    Returns
    -------
    EventStream
    """
    sim = DVSSimulator(threshold=threshold, gray_method=gray_method)
    return sim.simulate_video(video_path)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def render_event_frame(
    stream: EventStream,
    frame_idx: int,
    bg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render a single frame's events as a colour overlay.
    Positive events → red, negative events → blue.

    Parameters
    ----------
    stream    : EventStream
    frame_idx : int         Which frame to render.
    bg        : np.ndarray  Optional BGR background (H×W×3 uint8).

    Returns
    -------
    np.ndarray  BGR uint8 image.
    """
    H, W = stream.height, stream.width
    if bg is not None:
        canvas = bg.copy()
    else:
        canvas = np.full((H, W, 3), 20, dtype=np.uint8)  # dark background

    for e in stream.events:
        if e.t == frame_idx:
            color = (0, 0, 220) if e.polarity == 1 else (220, 0, 0)  # BGR
            cv2.circle(canvas, (e.x, e.y), 1, color, -1)

    return canvas


def save_event_video(
    stream: EventStream,
    output_path: str | Path,
    fps: Optional[float] = None,
) -> None:
    """Write the event visualisation to an MP4 file."""
    path   = Path(output_path)
    fps_   = fps or stream.fps
    H, W   = stream.height, stream.width
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps_, (W, H))

    for t in range(stream.n_frames):
        frame = render_event_frame(stream, t)
        writer.write(frame)

    writer.release()
    logger.info("Event video saved to %s", path)
