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

Events are stored as parallel NumPy arrays (structure-of-arrays) so that
simulation, encoding and rendering stay vectorised; ``EventStream.events``
builds ``Event`` objects lazily for convenience.

Reference:
    Rebecq et al., "ESIM: an Open Event Camera Simulator", CoRL 2018.
    Gallego et al., "Event-based Vision: A Survey", TPAMI 2022.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# One frame's worth of events: (xs, ys, polarities)
FrameEvents = Tuple[np.ndarray, np.ndarray, np.ndarray]


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


def _empty(dtype) -> np.ndarray:
    return np.zeros(0, dtype=dtype)


@dataclass
class EventStream:
    """
    Container for a sequence of events produced by the simulator.

    Events are kept sorted by frame index ``t`` (the simulator emits them in
    order), which lets ``frame_events`` slice a frame in O(log N).
    """
    x: np.ndarray = field(default_factory=lambda: _empty(np.int32))
    y: np.ndarray = field(default_factory=lambda: _empty(np.int32))
    t: np.ndarray = field(default_factory=lambda: _empty(np.int32))
    p: np.ndarray = field(default_factory=lambda: _empty(np.int8))
    width: int = 0
    height: int = 0
    n_frames: int = 0
    threshold: float = 0.2
    fps: float = 30.0

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def events(self) -> List[Event]:
        """Events as a list of ``Event`` objects (built on demand; slow for big streams)."""
        return [
            Event(int(x), int(y), int(t), int(p))
            for x, y, t, p in zip(self.x, self.y, self.t, self.p)
        ]

    def density(self) -> float:
        """Mean number of events per pixel per frame."""
        if self.n_frames == 0 or self.width == 0 or self.height == 0:
            return 0.0
        return len(self) / (self.n_frames * self.width * self.height)

    def to_numpy(self) -> np.ndarray:
        """Return Nx4 int32 array of (x, y, t, polarity)."""
        if len(self) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return np.stack([self.x, self.y, self.t, self.p.astype(np.int32)], axis=1).astype(np.int32)

    def frame_events(self, frame_idx: int) -> FrameEvents:
        """Return ``(xs, ys, polarities)`` of the events fired at ``frame_idx``."""
        lo = int(np.searchsorted(self.t, frame_idx, side="left"))
        hi = int(np.searchsorted(self.t, frame_idx, side="right"))
        return self.x[lo:hi], self.y[lo:hi], self.p[lo:hi]


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
        Lower  -> more events (finer detail, less compression).
        Higher -> fewer events (more compression, more error).
        Typical range: 0.05 - 0.5.  Default 0.2 is a good starting point.
    epsilon : float
        Small value added before log to avoid log(0).  Default 1e-6.
    gray_method : str
        How to convert BGR frames to grayscale.
        'cv2'  - fast OpenCV default.
        'luma' - BT.601 weighted (Y = 0.299R + 0.587G + 0.114B), closer
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
        if gray_method not in ("luma", "cv2"):
            raise ValueError(f"gray_method must be 'luma' or 'cv2', got {gray_method!r}")
        self.threshold = threshold
        self.epsilon = epsilon
        self.gray_method = gray_method

    # ------------------------------------------------------------------
    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 frame to float32 in [0, 1]."""
        if self.gray_method == "luma":
            f = frame_bgr.astype(np.float32)
            gray = 0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0]
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray / 255.0

    def _log_intensity(self, gray: np.ndarray) -> np.ndarray:
        return np.log(gray + self.epsilon)

    def log_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 frame -> log-intensity image."""
        return self._log_intensity(self._to_gray(frame_bgr))

    def step(self, prev_log: np.ndarray, curr_log: np.ndarray) -> FrameEvents:
        """Events fired between two consecutive log-intensity images."""
        diff = curr_log - prev_log
        ys, xs = np.nonzero(np.abs(diff) > self.threshold)
        pol = np.where(diff[ys, xs] > 0, 1, -1).astype(np.int8)
        return xs.astype(np.int32), ys.astype(np.int32), pol

    # ------------------------------------------------------------------
    def iter_events(self, frames: Iterator[np.ndarray]) -> Iterator[FrameEvents]:
        """
        Lazily convert an iterator of BGR frames into per-frame events.

        Yields exactly one ``FrameEvents`` per input frame; the first one is
        always empty (there is no previous frame to compare against).
        """
        prev_log: Optional[np.ndarray] = None
        for frame in frames:
            curr_log = self.log_frame(frame)
            if prev_log is None:
                yield (_empty(np.int32), _empty(np.int32), _empty(np.int8))
            else:
                yield self.step(prev_log, curr_log)
            prev_log = curr_log

    @staticmethod
    def _assemble(chunks: List[FrameEvents], width: int, height: int, n_frames: int,
                  threshold: float, fps: float) -> EventStream:
        if chunks:
            xs = np.concatenate([c[0] for c in chunks])
            ys = np.concatenate([c[1] for c in chunks])
            ps = np.concatenate([c[2] for c in chunks])
            ts = np.concatenate([np.full(len(c[0]), t, dtype=np.int32) for t, c in enumerate(chunks)])
        else:
            xs = ys = ts = _empty(np.int32)
            ps = _empty(np.int8)
        return EventStream(x=xs, y=ys, t=ts, p=ps, width=width, height=height,
                           n_frames=n_frames, threshold=threshold, fps=fps)

    # ------------------------------------------------------------------
    def simulate_array(self, frames: np.ndarray, fps: float = 30.0) -> EventStream:
        """
        Simulate events from a pre-loaded array of BGR frames.

        Parameters
        ----------
        frames : np.ndarray  shape (T, H, W, 3), dtype uint8
        fps    : float
        """
        T, H, W, _ = frames.shape
        chunks = list(self.iter_events(iter(frames)))
        stream = self._assemble(chunks, W, H, T, self.threshold, fps)
        logger.info(
            "Simulated %d events from %d frames (%dx%d) | density=%.4f",
            len(stream), T, W, H, stream.density(),
        )
        return stream

    # ------------------------------------------------------------------
    @staticmethod
    def open_video(video_path: str | Path) -> cv2.VideoCapture:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        return cap

    @staticmethod
    def iter_video_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            yield frame

    def simulate_video(self, video_path: str | Path) -> EventStream:
        """
        Simulate events frame-by-frame from a video file.

        Frames are streamed one at a time; only the (compact) events are kept
        in memory.  Suitable for long clips on consumer hardware.
        """
        cap = self.open_video(video_path)
        try:
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            chunks = list(self.iter_events(self.iter_video_frames(cap)))
        finally:
            cap.release()
        if not chunks:
            raise RuntimeError("Could not read first frame.")

        stream = self._assemble(chunks, width, height, len(chunks), self.threshold, fps)
        logger.info(
            "Simulated %d events from %s (%dx%d, %d frames, C=%.2f)",
            len(stream), Path(video_path).name, width, height, stream.n_frames, self.threshold,
        )
        return stream

    # ------------------------------------------------------------------
    def threshold_sweep(
        self, video_path: str | Path, thresholds: Optional[list] = None
    ) -> dict:
        """
        Sweep over multiple threshold values and return event counts.
        Useful for plotting event density vs C.
        """
        if thresholds is None:
            thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
        results = {}
        for c in thresholds:
            sim = DVSSimulator(threshold=c, epsilon=self.epsilon, gray_method=self.gray_method)
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
    Positive events -> red, negative events -> blue.

    Parameters
    ----------
    stream    : EventStream
    frame_idx : int         Which frame to render.
    bg        : np.ndarray  Optional BGR background (H x W x 3 uint8).

    Returns
    -------
    np.ndarray  BGR uint8 image.
    """
    H, W = stream.height, stream.width
    if bg is not None:
        canvas = bg.copy()
    else:
        canvas = np.full((H, W, 3), 20, dtype=np.uint8)  # dark background

    xs, ys, ps = stream.frame_events(frame_idx)
    pos = ps > 0
    canvas[ys[pos], xs[pos]] = (0, 0, 220)     # BGR red
    canvas[ys[~pos], xs[~pos]] = (220, 0, 0)   # BGR blue
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
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {path}")

    try:
        for t in range(stream.n_frames):
            writer.write(render_event_frame(stream, t))
    finally:
        writer.release()
    logger.info("Event video saved to %s", path)
