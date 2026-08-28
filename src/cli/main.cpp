// EchoScribe M0 skeleton CLI.
// Modes: --help, file transcription, real-time microphone (placeholders for now).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "audio/wav.hpp"
#include "encoder/encoder.hpp"
#include "cli/transcribe.hpp"
#include "frontend/logmel.hpp"
#include "gguf/gguf.hpp"
#include "gguf/loader.hpp"

namespace {

void printUsage(const char* prog) {
    std::printf(
        "EchoScribe - real-time speech recognition, from scratch in C++\n"
        "\n"
        "Usage:\n"
        "  %s <file.wav>              transcribe a 16kHz mono wav file\n"
        "  %s --mic [--lang LANG]     real-time microphone subtitles\n"
        "  %s --help                  show this help\n"
        "\n"
        "Options:\n"
        "  --model PATH        path to whisper GGUF model (default: models/whisper-tiny-F16.gguf)\n"
        "  --lang LANG         language hint, e.g. en (default: auto)\n"
        "  --list-tensors      print model metadata and all tensor names/shapes/dtypes\n"
        "  --encoder-stats     run the encoder on synthetic mel input, print activation stats\n"
        "\n",
        prog, prog, prog);
}

// Deterministic synthetic audio used for encoder sanity checks (mirrors the
// formula in the reference-generation script).
std::vector<float> syntheticAudio(size_t seconds) {
    std::vector<float> pcm(seconds * 16000);
    for (size_t i = 0; i < pcm.size(); ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        pcm[i] = static_cast<float>(0.3 * std::sin(2.0 * M_PI * 220.0 * t) +
                                    0.2 * std::sin(2.0 * M_PI * 445.0 * t) +
                                    0.15 * std::sin(2.0 * M_PI * 883.0 * t) * std::exp(-t / 2.0) +
                                    0.1 * std::sin(2.0 * M_PI * (300.0 + 50.0 * t) * t) *
                                        std::max(0.0, std::sin(M_PI * t / 2.0)));
    }
    return pcm;
}

void encoderStats(const std::string& modelPath) {
    gguf::File f = gguf::File::open(modelPath);
    whisper::Encoder enc(f);

    const auto pcm = syntheticAudio(10);
    tensor::Tensor mel = frontend::logMelSpectrogram(pcm.data(), pcm.size(), 160000);
    tensor::Tensor out = enc.forward(mel);

    std::printf("encoder: d_model=%zu n_head=%zu n_layers=%zu\n", enc.dModel(), enc.nHead(),
                enc.nLayers());
    std::printf("  input mel: %zux%zu -> output: %zux%zu\n", mel.shape()[0], mel.shape()[1],
                out.shape()[0], out.shape()[1]);

    double mean = 0, sq = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (size_t i = 0; i < out.size(); ++i) {
        mean += out.data()[i];
        sq += static_cast<double>(out.data()[i]) * out.data()[i];
        vmin = std::min(vmin, out.data()[i]);
        vmax = std::max(vmax, out.data()[i]);
    }
    mean /= out.size();
    const double stdv = std::sqrt(sq / out.size() - mean * mean);
    std::printf("  output stats: mean=%.6f std=%.6f min=%.4f max=%.4f\n", mean, stdv, vmin, vmax);
    std::printf("  token 0 (first 8): ");
    for (size_t j = 0; j < 8; ++j) std::printf("%.4f ", out.data()[j]);
    std::printf("\n  token %zu (first 8): ", out.shape()[0] - 1);
    for (size_t j = 0; j < 8; ++j) std::printf("%.4f ", out.data()[(out.shape()[0] - 1) * out.shape()[1] + j]);
    std::printf("\n");
}

void listTensors(const std::string& modelPath) {
    gguf::File f = gguf::File::open(modelPath);

    std::string arch = "?";
    if (f.hasKV("general.architecture")) {
        arch = std::get<std::string>(f.kv("general.architecture"));
    }
    std::printf("model: %s\n", modelPath.c_str());
    std::printf("  version=%u  kv=%zu  tensors=%llu  alignment=%llu  arch=%s\n",
                f.version(), f.kvCount(), static_cast<unsigned long long>(f.tensorCount()),
                static_cast<unsigned long long>(f.alignment()), arch.c_str());

    size_t maxName = 0;
    for (const gguf::TensorInfo& t : f.tensors()) maxName = std::max(maxName, t.name.size());

    size_t totalF32 = 0;
    for (const gguf::TensorInfo& t : f.tensors()) {
        std::string dims;
        for (size_t i = t.dims.size(); i-- > 0;) {
            dims += std::to_string(t.dims[i]);
            if (i != 0) dims += "x";
        }
        std::printf("  %-*s  %-6s %-20s", static_cast<int>(maxName), t.name.c_str(),
                    gguf::elementTypeName(t.type), dims.c_str());
        if (t.nBytes > 0) {
            std::printf(" %8.2f MB", static_cast<double>(t.nBytes) / (1024.0 * 1024.0));
            totalF32 += t.nBytes;
        } else {
            std::printf("     (quant)");
        }
        std::printf("\n");
    }
    std::printf("  total (f32/f16 tensors): %.1f MB\n",
                static_cast<double>(totalF32) / (1024.0 * 1024.0));

    // Load a tensor end-to-end (mmap + f16 -> f32) as a smoke check.
    if (f.findTensor("enc.conv.0.weight")) {
        tensor::Tensor w = gguf::loadTensorF32(f, "enc.conv.0.weight");
        std::printf("  smoke check: enc.conv.0.weight loaded, %zu elements, "
                    "first values %.4f %.4f %.4f\n",
                    w.size(), w.data()[0], w.data()[1], w.data()[2]);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string model = "models/whisper-tiny-F16.gguf";
    std::string lang = "auto";
    std::string inputFile;
    bool mic = false;
    bool listTensorsFlag = false;
    bool encoderStatsFlag = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--lang" && i + 1 < argc) {
            lang = argv[++i];
        } else if (arg == "--mic") {
            mic = true;
        } else if (arg == "--list-tensors") {
            listTensorsFlag = true;
        } else if (arg == "--encoder-stats") {
            encoderStatsFlag = true;
        } else if (!arg.empty() && arg[0] != '-') {
            inputFile = arg;
        } else {
            std::fprintf(stderr, "error: unknown or incomplete option '%s'\n\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    if (listTensorsFlag) {
        try {
            listTensors(model);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    if (encoderStatsFlag) {
        try {
            encoderStats(model);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    if (!inputFile.empty() && mic) {
        std::fprintf(stderr, "error: cannot combine file input with --mic\n");
        return 1;
    }

    if (mic) {
        std::printf("mic mode: not implemented yet (M7)\n");
        return 0;
    }
    if (!inputFile.empty()) {
        try {
            transcribe::Transcriber t(model);
            const auto segs = t.transcribeFile(inputFile, lang);
            for (const transcribe::Segment& s : segs) {
                if (s.text.empty()) continue;
                std::printf("[%02d:%06.3f --> %02d:%06.3f]  %s\n",
                            static_cast<int>(s.t0) / 60, std::fmod(s.t0, 60.0),
                            static_cast<int>(s.t1) / 60, std::fmod(s.t1, 60.0),
                            s.text.c_str());
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    printUsage(argv[0]);
    return argc > 1 ? 1 : 0;
}
