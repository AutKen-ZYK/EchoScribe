#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "audio/wav.hpp"
#include "cli/transcribe.hpp"
#include "streaming/stream.hpp"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#ifndef ECHOSCRIBE_MODELS_DIR
#define ECHOSCRIBE_MODELS_DIR "models"
#endif

using streaming::RingBuffer;
using streaming::Segmenter;
using streaming::TimedSegment;

TEST_CASE("ring buffer fifo semantics", "[streaming]") {
    RingBuffer rb(8);
    const float in1[] = {1, 2, 3, 4};
    const float in2[] = {5, 6, 7, 8, 9, 10};
    rb.write(in1, 4);
    rb.write(in2, 6); // overwrites oldest two

    float out[6] = {};
    REQUIRE(rb.size() == 8);
    REQUIRE(rb.read(out, 6) == 6);
    REQUIRE(out[0] == Catch::Approx(3.0f));
    REQUIRE(out[1] == Catch::Approx(4.0f));
    REQUIRE(out[5] == Catch::Approx(8.0f));
    REQUIRE(rb.read(out, 6) == 2); // remaining 9, 10
}

namespace {

// speech-like burst: sine with amplitude envelope, rms ≈ amp * 0.7
std::vector<float> speechBurst(double seconds, double amp, double freq = 220.0) {
    const size_t n = static_cast<size_t>(seconds * 16000);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        v[i] = static_cast<float>(amp * std::sin(2.0 * M_PI * freq * t));
    }
    return v;
}

std::vector<float> silence(double seconds) {
    return std::vector<float>(static_cast<size_t>(seconds * 16000), 0.0f);
}

} // namespace

TEST_CASE("segmenter hysteresis and pre-roll", "[streaming]") {
    Segmenter seg(16000);
    std::vector<TimedSegment> out;

    // 2 s silence, 3 s speech, 2 s silence, short blip, 1 s silence
    auto feedAll = [&](const std::vector<float>& v) { seg.feed(v.data(), v.size(), out); };
    feedAll(silence(2.0));
    REQUIRE(!seg.inSpeech());
    feedAll(speechBurst(3.0, 0.3));
    REQUIRE(seg.inSpeech());
    feedAll(silence(2.0));
    REQUIRE(!seg.inSpeech());
    REQUIRE(out.size() == 1);

    // pre-roll: segment must start ~400 ms before the speech onset (2.0 s)
    REQUIRE(out[0].t0 == Catch::Approx(2.0 - 0.4).margin(0.11));
    // segment length covers speech + pre-roll + trailing silence blocks
    REQUIRE(out[0].samples.size() > 3.0 * 16000);

    // a short blip below the min-segment length is discarded
    feedAll(speechBurst(0.2, 0.3));
    feedAll(silence(1.0));
    REQUIRE(out.size() == 1);
}

TEST_CASE("segmenter force-flushes over-long speech", "[streaming]") {
    Segmenter::Config cfg;
    cfg.maxSegmentSamples = 5 * 16000; // 5 s
    Segmenter seg(16000, cfg);
    std::vector<TimedSegment> out;

    seg.feed(speechBurst(12.0, 0.3).data(), 12 * 16000, out);
    REQUIRE(out.size() >= 2); // force-flushed into multiple segments
    REQUIRE(out[0].samples.size() <= 5.2 * 16000);
}

TEST_CASE("realtime loop on jfk samples: segments transcribe correctly", "[streaming][model]") {
    const std::string modelPath = ECHOSCRIBE_MODELS_DIR "/whisper-tiny-F16.gguf";
    bool haveModel = false;
    {
        std::ifstream f(modelPath, std::ios::binary | std::ios::ate);
        haveModel = f.good() && f.tellg() > 0;
    }
    if (!haveModel || !std::ifstream("tests/data/jfk.wav").good()) {
        WARN("model or jfk.wav missing, skipping realtime e2e test");
        return;
    }

    // simulate the mic loop: 1.5 s silence, jfk, 1.5 s silence, jfk, 1 s silence
    auto jfk = audio::readWav("tests/data/jfk.wav", 16000);
    std::vector<float> stream;
    auto append = [&](const std::vector<float>& v) { stream.insert(stream.end(), v.begin(), v.end()); };
    append(silence(1.5));
    append(jfk.samples);
    append(silence(1.5));
    append(jfk.samples);
    append(silence(1.0));

    transcribe::Transcriber t(modelPath);
    Segmenter seg(16000);
    std::vector<TimedSegment> out;

    // feed in 100 ms chunks like the capture callback would
    const size_t chunk = 1600;
    for (size_t off = 0; off < stream.size(); off += chunk) {
        seg.feed(stream.data() + off, std::min(chunk, stream.size() - off), out);
    }

    REQUIRE(out.size() == 2);
    // pre-roll guarantees segments start at or before the true onset; jfk.wav
    // has a soft onset, so the VAD may trigger slightly after 1.5 s
    REQUIRE(out[0].t0 == Catch::Approx(1.5 - 0.4).margin(0.5));
    REQUIRE(out[0].t0 <= 1.5);
    REQUIRE(out[1].t0 == Catch::Approx(1.5 + 11.0 + 1.5 - 0.4).margin(0.6));

    for (const auto& s : out) {
        auto parts = t.transcribeSegment(s.samples.data(), s.samples.size(), "en");
        bool ok = false;
        for (const auto& p : parts) {
            if (p.text.find("fellow Americans") != std::string::npos) ok = true;
        }
        INFO("segment at " << s.t0 << "s -> '"
                           << (parts.empty() ? std::string("(none)") : parts[0].text) << "'");
        REQUIRE(ok);
    }
}
