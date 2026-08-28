// EchoScribe M0 skeleton CLI.
// Modes: --help, file transcription, real-time microphone (placeholders for now).

#include <cstdio>
#include <cstring>
#include <string>

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
        "  --model PATH   path to whisper GGUF model (default: models/whisper-tiny.gguf)\n"
        "  --lang LANG    language hint, e.g. en (default: auto)\n"
        "\n",
        prog, prog, prog);
}

} // namespace

int main(int argc, char** argv) {
    std::string model = "models/whisper-tiny.gguf";
    std::string lang = "auto";
    std::string inputFile;
    bool mic = false;

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
        } else if (!arg.empty() && arg[0] != '-') {
            inputFile = arg;
        } else {
            std::fprintf(stderr, "error: unknown or incomplete option '%s'\n\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
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
