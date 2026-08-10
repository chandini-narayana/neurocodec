# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
Shared pytest fixtures for NeuroCodec tests.
"""

import numpy as np
import pytest


def make_moving_frames(height=32, width=32, n_frames=10) -> np.ndarray:
    """Alternating white/black checkerboard — maximum event activity."""
    frames = []
    for i in range(n_frames):
        f = np.zeros((height, width, 3), dtype=np.uint8)
        if i % 2 == 0:
            f[::2, ::2] = 255
        else:
            f[1::2, 1::2] = 255
        frames.append(f)
    return np.array(frames)


def make_static_frames(height=32, width=32, n_frames=10) -> np.ndarray:
    return np.zeros((n_frames, height, width, 3), dtype=np.uint8)


@pytest.fixture
def moving_frames():
    return make_moving_frames()


@pytest.fixture
def static_frames():
    return make_static_frames()


@pytest.fixture
def small_event_stream():
    """A small EventStream from moving frames, for encoder tests."""
    from neurocodec.simulator import DVSSimulator
    sim = DVSSimulator(threshold=0.1)
    return sim.simulate_array(make_moving_frames())
