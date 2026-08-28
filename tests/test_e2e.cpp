#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "audio/wav.hpp"
#include "cli/transcribe.hpp"

#include <fstream>
#include <string>
#include <vector>

#ifndef ECHOSCRIBE_MODELS_DIR
#define ECHOSCRIBE_MODELS_DIR "models"
#endif

using Catch::Approx;
using transcribe::Segment;
using transcribe::Transcriber;

namespace {

bool fileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() && f.tellg() > 0;
}

} // namespace

// End-to-end: wav file -> text with timestamps. Reference behavior comes from
// the official whisper tiny model (see tests/data). Skipped when data missing.
TEST_CASE("end-to-end file transcription", "[e2e][model]") {
    const std::string modelPath = ECHOSCRIBE_MODELS_DIR "/whisper-tiny-F16.gguf";
    if (!fileExists(modelPath) || !fileExists("tests/data/jfk.wav")) {
        WARN("model or jfk.wav missing, skipping e2e test");
        return;
    }

    Transcriber t(modelPath);
    const auto segs = t.transcribeFile("tests/data/jfk.wav", "en");

    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].t0 == Approx(0.0).margin(1e-6));
    REQUIRE(segs[0].t1 == Approx(10.5).margin(1e-6));
    REQUIRE(segs[0].text ==
            "And so my fellow Americans ask not what your country can do for you, ask what you can "
            "do for your country.");
}

TEST_CASE("end-to-end: silence produces no hallucinated segments", "[e2e][model]") {
    const std::string modelPath = ECHOSCRIBE_MODELS_DIR "/whisper-tiny-F16.gguf";
    if (!fileExists(modelPath)) {
        WARN("model missing, skipping silence test");
        return;
    }

    Transcriber t(modelPath);
    std::vector<float> silence(30 * 16000, 0.0f);
    const auto segs = t.transcribe(silence.data(), silence.size(), "en");
    INFO("segments from silence: " << segs.size());
    for (const auto& s : segs) INFO("  '" << s.text << "'");
    REQUIRE(segs.empty());
}

TEST_CASE("end-to-end: long audio spans multiple 30s windows", "[e2e][model]") {
    const std::string modelPath = ECHOSCRIBE_MODELS_DIR "/whisper-tiny-F16.gguf";
    if (!fileExists(modelPath) || !fileExists("tests/data/jfk.wav")) {
        WARN("model or jfk.wav missing, skipping multi-window test");
        return;
    }

    // jfk (11 s) + 25 s silence + jfk again = 47 s -> two 30 s windows
    auto jfk = audio::readWav("tests/data/jfk.wav", 16000);
    std::vector<float> pcm(47 * 16000, 0.0f);
    std::copy(jfk.samples.begin(), jfk.samples.end(), pcm.begin());
    std::copy(jfk.samples.begin(), jfk.samples.end(), pcm.begin() + 36 * 16000);

    Transcriber t(modelPath);
    const auto segs = t.transcribe(pcm.data(), pcm.size(), "en");

    REQUIRE(segs.size() >= 2);
    // first window: the quote at ~0 s
    REQUIRE(segs[0].t0 == Approx(0.0).margin(1e-6));
    REQUIRE(segs[0].text.find("fellow Americans") != std::string::npos);
    // second window: quote again; window-relative timestamps start near 0
    // (whisper's max_initial_timestamp rule forces the first timestamp <= 1 s,
    // so with leading silence inside the window the absolute start is ~30 s)
    bool foundSecond = false;
    for (const auto& s : segs) {
        if (s.t0 >= 30.0 && s.t0 < 38.0 && s.text.find("fellow Americans") != std::string::npos) {
            foundSecond = true;
        }
    }
    REQUIRE(foundSecond);
}
