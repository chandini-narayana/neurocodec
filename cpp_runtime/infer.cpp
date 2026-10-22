// Copyright (c) 2026 Chandini Narayana
// SPDX-License-Identifier: MIT
/**
 * NeuroCodec C++ ONNX Runtime Inference
 * =======================================
 * Decodes a .nec file (format v1 or v2) to raw grayscale frames using the
 * decoder exported by eval/export_onnx.py (decoder.onnx).
 *
 * The exported graph is a single recurrent step: inputs `voxel`, `h0`, `c0`,
 * `h1`, `c1`, ... and outputs `frame`, `h0_out`, `c0_out`, ...  This program
 * owns the ConvLSTM state and feeds it back after every frame, exactly like
 * the PyTorch decoder does.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && cmake --build . --config Release
 *
 * Usage:
 *   ./nc_decode --model decoder.onnx --input clip.nec --output frames/
 *   ./nc_decode --model decoder.onnx --input clip.nec --benchmark
 *
 * Output: frames/frame_000001.pgm, frame_000002.pgm, ...
 *         (PGM = portable graymap, viewable in any image viewer)
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

// ── .nec reader (v1 raw, v2 DEFLATE) ──────────────────────────────────────
// See neurocodec/encoder.py for the format description.

struct NECHeader {
    uint16_t width    = 0;
    uint16_t height   = 0;
    uint8_t  n_bins   = 0;
    float    threshold = 0.f;
    uint32_t n_frames = 0;
    float    fps      = 30.f;
    int      version  = 0;
};

struct SparseFrame {
    std::vector<int32_t> indices;
    std::vector<float>   values;
};

class NECReader {
public:
    explicit NECReader(const std::string& path) {
        file_.open(path, std::ios::binary);
        if (!file_)
            throw std::runtime_error("Cannot open: " + path);

        char magic[4];
        read_exact(magic, 4);
        if (std::memcmp(magic, "NEC\x02", 4) == 0)      hdr_.version = 2;
        else if (std::memcmp(magic, "NEC\x01", 4) == 0) hdr_.version = 1;
        else throw std::runtime_error("Not a .nec file (bad magic)");

        read_exact(&hdr_.width,     2);
        read_exact(&hdr_.height,    2);
        read_exact(&hdr_.n_bins,    1);
        read_exact(&hdr_.threshold, 4);
        read_exact(&hdr_.n_frames,  4);
        if (hdr_.version == 2) read_exact(&hdr_.fps, 4);

        if (hdr_.width == 0 || hdr_.height == 0 || hdr_.n_bins == 0)
            throw std::runtime_error("Invalid .nec header (zero dimension)");
        n_cells_ = static_cast<size_t>(hdr_.n_bins) * hdr_.height * hdr_.width;
    }

    const NECHeader& header() const { return hdr_; }

    bool next_frame(SparseFrame& out) {
        if (frames_read_ >= hdr_.n_frames) return false;

        uint32_t len = 0;
        file_.read(reinterpret_cast<char*>(&len), 4);
        if (!file_) return false;

        if (hdr_.version == 1) {
            const uint32_t n = len;  // v1: record starts with the pair count
            if (n > n_cells_) throw std::runtime_error("Corrupt .nec frame");
            out.indices.resize(n);
            out.values.resize(n);
            read_exact(out.indices.data(), static_cast<size_t>(n) * 4);
            read_exact(out.values.data(),  static_cast<size_t>(n) * 4);
        } else {
            std::vector<Bytef> comp(len);
            read_exact(comp.data(), len);
            // raw = u32 n | 4n bytes of index-delta byte planes | n int8 values
            std::vector<Bytef> raw(4 + 5 * n_cells_);
            uLongf raw_len = static_cast<uLongf>(raw.size());
            if (uncompress(raw.data(), &raw_len, comp.data(), len) != Z_OK)
                throw std::runtime_error("Corrupt .nec frame (zlib)");
            uint32_t n = 0;
            if (raw_len < 4) throw std::runtime_error("Corrupt .nec frame");
            std::memcpy(&n, raw.data(), 4);
            if (n > n_cells_ || raw_len != 4 + 5ull * n)
                throw std::runtime_error("Corrupt .nec frame (length)");

            out.indices.resize(n);
            out.values.resize(n);
            const Bytef* planes = raw.data() + 4;
            const int8_t* q = reinterpret_cast<const int8_t*>(planes + 4ull * n);
            uint64_t idx = 0;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t d = static_cast<uint32_t>(planes[i])
                           | (static_cast<uint32_t>(planes[n + i])     << 8)
                           | (static_cast<uint32_t>(planes[2 * n + i]) << 16)
                           | (static_cast<uint32_t>(planes[3 * n + i]) << 24);
                idx += d;
                if (idx >= n_cells_) throw std::runtime_error("Corrupt .nec frame (index)");
                out.indices[i] = static_cast<int32_t>(idx);
                out.values[i]  = static_cast<float>(q[i]) / 127.0f;
            }
        }
        ++frames_read_;
        return true;
    }

    // Reconstruct one dense voxel [B*H*W] from a sparse frame
    std::vector<float> to_dense(const SparseFrame& sf) const {
        std::vector<float> grid(n_cells_, 0.0f);
        for (size_t i = 0; i < sf.indices.size(); ++i) {
            const auto k = static_cast<size_t>(sf.indices[i]);
            if (k >= n_cells_) throw std::runtime_error("Corrupt .nec frame (index)");
            grid[k] = sf.values[i];
        }
        return grid;
    }

private:
    void read_exact(void* dst, size_t n) {
        file_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (static_cast<size_t>(file_.gcount()) != n)
            throw std::runtime_error("Truncated .nec file");
    }

    std::ifstream file_;
    NECHeader     hdr_{};
    size_t        n_cells_ = 0;
    uint32_t      frames_read_ = 0;
};

// ── PGM writer (no external deps) ────────────────────────────────────────

void write_pgm(const std::string& path,
               const float* data, int H, int W) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f << "P5\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < H * W; ++i) {
        uint8_t px = static_cast<uint8_t>(
            std::clamp(data[i] * 255.0f + 0.5f, 0.0f, 255.0f));
        f.put(static_cast<char>(px));
    }
}

// ── Simple arg parser ─────────────────────────────────────────────────────

struct Args {
    std::string model_path;
    std::string input_path;
    std::string output_dir = "frames";
    bool        benchmark  = false;
    bool        help       = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model"  && i + 1 < argc) a.model_path  = argv[++i];
        else if (arg == "--input"  && i + 1 < argc) a.input_path  = argv[++i];
        else if (arg == "--output" && i + 1 < argc) a.output_dir  = argv[++i];
        else if (arg == "--benchmark") a.benchmark = true;
        else if (arg == "--help" || arg == "-h") a.help = true;
    }
    return a;
}

// ── Main ─────────────────────────────────────────────────────────────────

int run(const Args& args) {
    // ── Load .nec ────────────────────────────────────────────────────────
    std::cout << "Loading " << args.input_path << "...\n";
    NECReader reader(args.input_path);
    const auto& hdr = reader.header();
    std::cout << "  Format:     v" << hdr.version << "\n"
              << "  Resolution: " << hdr.width << "x" << hdr.height << "\n"
              << "  Bins:       " << static_cast<int>(hdr.n_bins) << "\n"
              << "  Frames:     " << hdr.n_frames << "\n"
              << "  Threshold:  " << hdr.threshold << "\n\n";

    // ── Initialise ONNX Runtime ───────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "neurocodec");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading model " << args.model_path << "...\n";
#ifdef _WIN32
    const std::wstring wmodel(args.model_path.begin(), args.model_path.end());
    Ort::Session session(env, wmodel.c_str(), opts);
#else
    Ort::Session session(env, args.model_path.c_str(), opts);
#endif
    Ort::AllocatorWithDefaultOptions alloc;

    const size_t n_in  = session.GetInputCount();
    const size_t n_out = session.GetOutputCount();
    if (n_in < 1 || n_in % 2 != 1 || n_out != n_in)
        throw std::runtime_error(
            "Unexpected model signature: expected voxel + (h,c) state pairs. "
            "Re-export with eval/export_onnx.py");

    std::vector<std::string> in_names, out_names;
    for (size_t i = 0; i < n_in; ++i)
        in_names.push_back(session.GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < n_out; ++i)
        out_names.push_back(session.GetOutputNameAllocated(i, alloc).get());
    std::vector<const char*> in_ptrs, out_ptrs;
    for (auto& s : in_names)  in_ptrs.push_back(s.c_str());
    for (auto& s : out_names) out_ptrs.push_back(s.c_str());
    std::cout << "  Inputs:  " << n_in << " (voxel + " << (n_in - 1) / 2 << " ConvLSTM layers)\n\n";

    const int64_t B = hdr.n_bins;
    const int64_t H = hdr.height;
    const int64_t W = hdr.width;

    // ── Recurrent state (zero-initialised), channels read from the model ──
    std::vector<std::vector<float>> state(n_in - 1);
    std::vector<std::vector<int64_t>> state_shape(n_in - 1);
    for (size_t i = 1; i < n_in; ++i) {
        auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[1] <= 0)
            throw std::runtime_error("Unexpected state tensor shape in model");
        state_shape[i - 1] = {1, shape[1], H, W};
        state[i - 1].assign(static_cast<size_t>(shape[1] * H * W), 0.0f);
    }

    if (!args.benchmark)
        fs::create_directories(args.output_dir);

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::vector<int64_t> voxel_shape = {1, B, H, W};

    std::vector<double> latencies;
    latencies.reserve(hdr.n_frames);

    SparseFrame sf;
    uint32_t frame_idx = 0;

    while (reader.next_frame(sf)) {
        auto dense = reader.to_dense(sf);                    // [B*H*W]

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, dense.data(), dense.size(),
            voxel_shape.data(), voxel_shape.size()));
        for (size_t i = 0; i < state.size(); ++i)
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info, state[i].data(), state[i].size(),
                state_shape[i].data(), state_shape[i].size()));

        auto t0 = std::chrono::high_resolution_clock::now();
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            in_ptrs.data(),  inputs.data(), inputs.size(),
            out_ptrs.data(), out_ptrs.size());
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);

        // Feed the updated ConvLSTM state back for the next frame
        for (size_t i = 0; i < state.size(); ++i) {
            const float* src = outputs[i + 1].GetTensorData<float>();
            std::copy(src, src + state[i].size(), state[i].begin());
        }

        if (!args.benchmark) {
            // outputs[0] shape: (1, 1, H, W)
            const float* out_data = outputs[0].GetTensorData<float>();

            std::ostringstream oss;
            oss << args.output_dir << "/frame_"
                << std::setw(6) << std::setfill('0') << (frame_idx + 1)
                << ".pgm";
            write_pgm(oss.str(), out_data,
                      static_cast<int>(H), static_cast<int>(W));
        }

        if ((frame_idx + 1) % 10 == 0 || frame_idx == 0)
            std::cout << "  Frame " << (frame_idx + 1) << "/" << hdr.n_frames
                      << "  (" << std::fixed << std::setprecision(2) << ms << " ms)\n";

        ++frame_idx;
    }

    if (frame_idx != hdr.n_frames)
        std::cerr << "warning: file declares " << hdr.n_frames
                  << " frames but only " << frame_idx << " could be read\n";

    // ── Summary ──────────────────────────────────────────────────────────
    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        double mean  = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                       / latencies.size();
        double p50   = latencies[latencies.size() * 50 / 100];
        double p99   = latencies[std::min(latencies.size() - 1, latencies.size() * 99 / 100)];
        double total = std::accumulate(latencies.begin(), latencies.end(), 0.0);

        std::cout << "\n-- Benchmark (model run only) -------------------\n"
                  << "  Frames decoded : " << frame_idx << "\n"
                  << "  Mean latency   : " << std::fixed << std::setprecision(2)
                  << mean  << " ms\n"
                  << "  P50 latency    : " << p50   << " ms\n"
                  << "  P99 latency    : " << p99   << " ms\n"
                  << "  Total time     : " << total / 1000.0 << " s\n"
                  << "  Throughput     : "
                  << std::setprecision(1) << frame_idx / (total / 1000.0)
                  << " fps\n"
                  << "-------------------------------------------------\n";
    }

    if (!args.benchmark)
        std::cout << "\nFrames written to " << args.output_dir << "/\n";

    return 0;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.help || args.model_path.empty() || args.input_path.empty()) {
        std::cerr
            << "NeuroCodec C++ ONNX Decoder\n\n"
            << "Usage:\n"
            << "  nc_decode --model decoder.onnx --input clip.nec [--output frames/]\n"
            << "  nc_decode --model decoder.onnx --input clip.nec --benchmark\n\n"
            << "Options:\n"
            << "  --model    Path to exported decoder ONNX file (eval/export_onnx.py)\n"
            << "  --input    Path to .nec encoded video\n"
            << "  --output   Output directory for PGM frames (default: frames/)\n"
            << "  --benchmark  Print per-frame latency, skip writing output\n";
        return args.help ? 0 : 1;
    }

    try {
        return run(args);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
