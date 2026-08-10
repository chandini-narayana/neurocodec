# NeuroCodec C++ ONNX Runtime

CPU-native decoder that runs `decoder.onnx` without Python.
Achieves 2–5× faster inference than the PyTorch path.

## Prerequisites

- CMake ≥ 3.18
- C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC 2019)
- Internet connection (CMake fetches ONNX Runtime automatically)

## Build

```bash
cd cpp_runtime
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j4
```

The binary is at `build/nc_decode` (Linux/Mac) or `build/Release/nc_decode.exe` (Windows).

## Export the model from Python first

```python
from neurocodec.decoder import ConvLSTMDecoder
import torch

model = ConvLSTMDecoder(n_bins=5)
# load weights: model.load_state_dict(torch.load("decoder_v1.pt")["model_state_dict"])
model.eval()

dummy = torch.zeros(1, 5, 240, 320)   # batch=1, bins=5, H=240, W=320
torch.onnx.export(
    model, dummy, "decoder.onnx",
    input_names=["voxel"],
    output_names=["frame"],
    dynamic_axes={"voxel": {0: "batch", 2: "H", 3: "W"},
                  "frame": {0: "batch", 2: "H", 3: "W"}},
    opset_version=17,
)
print("Exported decoder.onnx")
```

## Usage

```bash
# Decode .nec → PGM frames
./build/nc_decode --model decoder.onnx --input clip.nec --output frames/

# Benchmark latency (no output written)
./build/nc_decode --model decoder.onnx --input clip.nec --benchmark
```

## Expected output

```
Loading clip.nec...
  Resolution: 640x480
  Bins:       5
  Frames:     150
  Threshold:  0.20

── Benchmark ──────────────────────────────────
  Frames decoded : 150
  Mean latency   : 34.2 ms
  P50 latency    : 33.8 ms
  P99 latency    : 41.1 ms
  Throughput     : 29.2 fps
────────────────────────────────────────────────
```
