# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""Tests for DVS event simulator."""

import pytest

from neurocodec.simulator import DVSSimulator, EventStream, render_event_frame
from tests.conftest import make_moving_frames, make_static_frames


class TestDVSSimulator:
    def test_static_frames_produce_no_events(self):
        sim    = DVSSimulator(threshold=0.2)
        frames = make_static_frames()
        stream = sim.simulate_array(frames)
        assert len(stream) == 0, "Static frames should produce zero events"

    def test_moving_frames_produce_events(self):
        sim    = DVSSimulator(threshold=0.1)
        frames = make_moving_frames()
        stream = sim.simulate_array(frames)
        assert len(stream) > 0, "Moving frames must produce events"

    def test_event_stream_metadata(self):
        sim    = DVSSimulator(threshold=0.2)
        frames = make_moving_frames(height=16, width=24, n_frames=5)
        stream = sim.simulate_array(frames)
        assert stream.height == 16
        assert stream.width  == 24

    def test_lower_threshold_more_events(self):
        frames  = make_moving_frames()
        sim_hi  = DVSSimulator(threshold=0.5)
        sim_lo  = DVSSimulator(threshold=0.05)
        high_c  = sim_hi.simulate_array(frames)
        low_c   = sim_lo.simulate_array(frames)
        assert len(low_c) >= len(high_c), "Lower threshold must produce >= events"

    def test_polarity_values(self):
        sim    = DVSSimulator(threshold=0.05)
        frames = make_moving_frames()
        stream = sim.simulate_array(frames)
        polarities = {e.polarity for e in stream.events}
        assert polarities.issubset({-1, 1}), "Polarity must be +1 or -1"

    def test_to_numpy_shape(self):
        sim    = DVSSimulator(threshold=0.1)
        frames = make_moving_frames()
        stream = sim.simulate_array(frames)
        arr    = stream.to_numpy()
        assert arr.ndim  == 2
        assert arr.shape[1] == 4  # x, y, t, polarity

    def test_event_density_in_range(self):
        sim    = DVSSimulator(threshold=0.2)
        frames = make_moving_frames()
        stream = sim.simulate_array(frames)
        d      = stream.density()
        assert 0.0 <= d <= 1.0

    def test_render_event_frame_shape(self):
        sim    = DVSSimulator(threshold=0.1)
        frames = make_moving_frames(height=32, width=32)
        stream = sim.simulate_array(frames)
        img    = render_event_frame(stream, frame_idx=1)
        assert img.shape == (32, 32, 3)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            DVSSimulator(threshold=0.0)
        with pytest.raises(ValueError):
            DVSSimulator(threshold=-0.5)

    def test_gray_method_luma(self):
        sim    = DVSSimulator(threshold=0.1, gray_method="luma")
        frames = make_moving_frames()
        stream = sim.simulate_array(frames)
        assert len(stream) > 0

    def test_gray_method_cv2(self):
        sim    = DVSSimulator(threshold=0.1, gray_method="cv2")
        frames = make_moving_frames()
        stream = sim.simulate_array(frames)
        assert len(stream) > 0

    def test_to_numpy_empty_stream(self):
        stream = EventStream(width=32, height=32, n_frames=5, threshold=0.2)
        arr    = stream.to_numpy()
        assert arr.shape == (0, 4)

    def test_event_coordinates_in_bounds(self):
        sim    = DVSSimulator(threshold=0.05)
        H, W   = 32, 48
        frames = make_moving_frames(height=H, width=W)
        stream = sim.simulate_array(frames)
        for e in stream.events:
            assert 0 <= e.x < W, f"x={e.x} out of bounds [0,{W})"
            assert 0 <= e.y < H, f"y={e.y} out of bounds [0,{H})"
