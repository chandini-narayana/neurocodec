// Copyright (c) 2026 Chandini Narayana
// SPDX-License-Identifier: MIT
/**
 * NeuroCodec C++ ONNX Runtime Inference
 * =======================================
 * Decodes a .nec file to raw grayscale frames using the exported
 * ConvLSTM decoder (decoder.onnx).
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
#include <cassert>
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

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

// ── .nec reader ───────────────────────────────────────────────────────────

static constexpr uint8_t NEC_MAGIC[4] = {'N', 'E', 'C', 0x01};

struct NECHeader {
    uint16_t width;
    uint16_t height;
    uint8_t  n_bins;
    float    threshold;
    uint32_t n_frames;
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

        uint8_t magic[4];
        file_.read(reinterpret_cast<char*>(magic), 4);
        for (int i = 0; i < 4; ++i)
            if (magic[i] != NEC_MAGIC[i])
                throw std::runtime_error("Not a .nec file (bad magic)");

        file_.read(reinterpret_cast<char*>(&hdr_.width),    2);
        file_.read(reinterpret_cast<char*>(&hdr_.height),   2);
        file_.read(reinterpret_cast<char*>(&hdr_.n_bins),   1);
        file_.read(reinterpret_cast<char*>(&hdr_.threshold), 4);
        file_.read(reinterpret_cast<char*>(&hdr_.n_frames), 4);
    }

    const NECHeader& header() const { return hdr_; }

    bool next_frame(SparseFrame& out) {
        if (frames_read_ >= hdr_.n_frames) return false;

        uint32_t n = 0;
        file_.read(reinterpret_cast<char*>(&n), 4);
        if (!file_) return false;

        out.indices.resize(n);
        out.values.resize(n);
        file_.read(reinterpret_cast<char*>(out.indices.data()), n * 4);
        file_.read(reinterpret_cast<char*>(out.values.data()),  n * 4);
        ++frames_read_;
        return file_.good() || n == 0;
    }

    // Reconstruct one dense voxel [B*H*W] from a sparse frame
    std::vector<float> to_dense(const SparseFrame& sf) const {
        const size_t total = static_cast<size_t>(hdr_.n_bins)
                           * hdr_.height * hdr_.width;
        std::vector<float> grid(total, 0.0f);
        for (size_t i = 0; i < sf.indices.size(); ++i)
            grid[static_cast<size_t>(sf.indices[i])] = sf.values[i];
        return grid;
    }

private:
    std::ifstream file_;
    NECHeader     hdr_{};
    uint32_t      frames_read_ = 0;
};

// ── PGM writer (no external deps) ────────────────────────────────────────

void write_pgm(const std::string& path,
               const float* data, int H, int W) {
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < H * W; ++i) {
        uint8_t px = static_cast<uint8_t>(
            std::clamp(data[i] * 255.0f, 0.0f, 255.0f));
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

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.help || args.model_path.empty() || args.input_path.empty()) {
        std::cerr
            << "NeuroCodec C++ ONNX Decoder\n\n"
            << "Usage:\n"
            << "  nc_decode --model decoder.onnx --input clip.nec [--output frames/]\n"
            << "  nc_decode --model decoder.onnx --input clip.nec --benchmark\n\n"
            << "Options:\n"
            << "  --model    Path to exported decoder ONNX file\n"
            << "  --input    Path to .nec encoded video\n"
            << "  --output   Output directory for PGM frames (default: frames/)\n"
            << "  --benchmark  Print per-frame latency, skip writing output\n";
        return args.help ? 0 : 1;
    }

    // ── Load .nec ────────────────────────────────────────────────────────
    std::cout << "Loading " << args.input_path << "...\n";
    NECReader reader(args.input_path);
    const auto& hdr = reader.header();
    std::cout << "  Resolution: " << hdr.width << "x" << hdr.height << "\n"
              << "  Bins:       " << static_cast<int>(hdr.n_bins) << "\n"
              << "  Frames:     " << hdr.n_frames << "\n"
              << "  Threshold:  " << hdr.threshold << "\n\n";

    // ── Initialise ONNX Runtime ───────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "neurocodec");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading model " << args.model_path << "...\n";
    Ort::Session session(env, args.model_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;

    // Get input/output names
    std::string in_name  = session.GetInputNameAllocated(0, alloc).get();
    std::string out_name = session.GetOutputNameAllocated(0, alloc).get();
    std::cout << "  Input:  " << in_name  << "\n"
              << "  Output: " << out_name << "\n\n";

    // ── Prepare output dir ───────────────────────────────────────────────
    if (!args.benchmark)
        fs::create_directories(args.output_dir);

    // ── Decode loop ──────────────────────────────────────────────────────
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const int64_t B  = hdr.n_bins;
    const int64_t H  = hdr.height;
    const int64_t W  = hdr.width;
    const size_t  voxel_size = static_cast<size_t>(B * H * W);

    std::vector<double> latencies;
    latencies.reserve(hdr.n_frames);

    SparseFrame sf;
    uint32_t frame_idx = 0;

    while (reader.next_frame(sf)) {
        auto dense = reader.to_dense(sf);                    // [B*H*W]

        // Shape: (1, B, H, W)  — batch=1
        std::vector<int64_t> input_shape = {1, B, H, W};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            dense.data(), dense.size(),
            input_shape.data(), input_shape.size());

        const char* in_names[]  = {in_name.c_str()};
        const char* out_names[] = {out_name.c_str()};

        auto t0 = std::chrono::high_resolution_clock::now();
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            in_names,  &input_tensor, 1,
            out_names, 1);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);

        if (!args.benchmark) {
            // outputs[0] shape: (1, 1, H, W)
            float* out_data = outputs[0].GetTensorMutableData<float>();

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

    // ── Summary ──────────────────────────────────────────────────────────
    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        double mean  = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                       / latencies.size();
        double p50   = latencies[latencies.size() * 50 / 100];
        double p99   = latencies[latencies.size() * 99 / 100];
        double total = std::accumulate(latencies.begin(), latencies.end(), 0.0);

        std::cout << "\n── Benchmark ──────────────────────────────────\n"
                  << "  Frames decoded : " << frame_idx << "\n"
                  << "  Mean latency   : " << std::fixed << std::setprecision(2)
                  << mean  << " ms\n"
                  << "  P50 latency    : " << p50   << " ms\n"
                  << "  P99 latency    : " << p99   << " ms\n"
                  << "  Total time     : " << total / 1000.0 << " s\n"
                  << "  Throughput     : "
                  << std::setprecision(1) << frame_idx / (total / 1000.0)
                  << " fps\n"
                  << "────────────────────────────────────────────────\n";
    }

    if (!args.benchmark)
        std::cout << "\nFrames written to " << args.output_dir << "/\n";

    return 0;
}
