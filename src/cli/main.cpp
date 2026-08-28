// EchoScribe M0 skeleton CLI.
// Modes: --help, file transcription, real-time microphone (placeholders for now).

#include <cstdio>
#include <cstring>
#include <string>

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
        "\n",
        prog, prog, prog);
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

    if (!inputFile.empty() && mic) {
        std::fprintf(stderr, "error: cannot combine file input with --mic\n");
        return 1;
    }

    if (mic) {
        std::printf("mic mode: not implemented yet (M7)\n");
        return 0;
    }
    if (!inputFile.empty()) {
        std::printf("file mode: '%s' not implemented yet (M6)\n", inputFile.c_str());
        return 0;
    }

    printUsage(argv[0]);
    return argc > 1 ? 1 : 0;
}
