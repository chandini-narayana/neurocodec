# NeuroCodec C++ ONNX Runtime

CPU-native decoder that runs the exported `decoder.onnx` without Python. Reads `.nec`
v1 and v2 files (v2 needs zlib). **Not yet compiled or benchmarked in CI**; no speed-up
figure is claimed until it has been measured.

## Prerequisites

- CMake ≥ 3.18, C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC 2019), zlib development files
- Either internet access (CMake downloads the official prebuilt ONNX Runtime for your
  platform) or an existing install passed as `-DORT_ROOT=/path/to/onnxruntime`

## Export the model from Python first

The export is a *single recurrent step*: inputs `voxel`, `h0`, `c0`, `h1`, `c1`, …, outputs
`frame`, `h0_out`, `c0_out`, …. `nc_decode` keeps the ConvLSTM state and feeds it back
each frame, so results match the PyTorch decoder.

```bash
pip install -e ".[onnx]"
python eval/export_onnx.py --weights decoder_v1.pt --output decoder.onnx --height 270 --width 480
```

The script also checks the ONNX graph against PyTorch over three stateful frames.

## Build

```bash
cd cpp_runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

The binary is `build/nc_decode` (Linux/macOS) or `build/Release/nc_decode.exe` (Windows).

## Usage

```bash
./build/nc_decode --model decoder.onnx --input clip.nec --output frames/   # PGM frames
./build/nc_decode --model decoder.onnx --input clip.nec --benchmark        # latency only
```

`--benchmark` reports the ONNX model run time per frame (not .nec parsing or PGM writing).
