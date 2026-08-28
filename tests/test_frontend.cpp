#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "frontend/logmel.hpp"
#include "frontend/mel.hpp"
#include "gguf/gguf.hpp"
#include "gguf/loader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;

#ifndef ECHOSCRIBE_MODELS_DIR
#define ECHOSCRIBE_MODELS_DIR "models"
#endif

namespace {

bool modelExists(std::string& path) {
    const char* names[] = {ECHOSCRIBE_MODELS_DIR "/whisper-tiny-F16.gguf"};
    for (const char* n : names) {
        std::ifstream f(n, std::ios::binary | std::ios::ate);
        if (f.good() && f.tellg() > 0) {
            path = n;
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("hann window properties", "[frontend]") {
    const auto w = mel::hannWindow(400);
    REQUIRE(w.size() == 400);
    REQUIRE_THAT(w[0], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(w[200], WithinAbs(1.0, 1e-6));
    // periodic hann sums to n/2 exactly
    double sum = 0;
    for (float v : w) sum += v;
    REQUIRE(sum == Catch::Approx(200.0).margin(1e-3));
    // symmetry: w[k] == w[n - k]
    REQUIRE_THAT(w[1], WithinAbs(w[399], 1e-6));
}

TEST_CASE("mel filterbank structure", "[frontend]") {
    constexpr size_t nMels = 80, nFft = 400, sr = 16000;
    const auto fb = mel::filterbank(nMels, nFft, sr);
    REQUIRE(fb.size() == nMels * (nFft / 2 + 1));

    // non-negative everywhere
    for (float v : fb) REQUIRE(v >= 0.0f);

    // each filter peaks inside its support; support is contiguous
    for (size_t m = 0; m < nMels; ++m) {
        const float* row = fb.data() + m * (nFft / 2 + 1);
        int nzStart = -1, nzEnd = -1;
        float peak = 0;
        for (size_t k = 0; k <= nFft / 2; ++k) {
            if (row[k] > 0 && nzStart < 0) nzStart = static_cast<int>(k);
            if (row[k] > 0) nzEnd = static_cast<int>(k);
            peak = std::max(peak, row[k]);
        }
        REQUIRE(nzStart >= 0);
        // Low-mel filters can be narrower than one FFT bin spacing (40 Hz),
        // leaving a single nonzero bin: librosa's table has 391 nonzeros total.
        REQUIRE(nzEnd >= nzStart);
        REQUIRE(peak > 0.0f);
        // slaney norm in Hz: peak = 2 / (base width in Hz) <= 2/75.4
        REQUIRE(peak < 0.03f);
        // center of mass roughly monotone across filters
        if (m > 0) {
            const float* prev = fb.data() + (m - 1) * (nFft / 2 + 1);
            double cmPrev = 0, cmCur = 0, wPrev = 0, wCur = 0;
            for (size_t k = 0; k <= nFft / 2; ++k) {
                cmPrev += prev[k] * k; wPrev += prev[k];
                cmCur += row[k] * k;   wCur += row[k];
            }
            REQUIRE(cmCur / wCur > cmPrev / wPrev);
        }
    }
}

TEST_CASE("logmel: pure silence normalizes to -1.5", "[frontend]") {
    std::vector<float> silence(480000, 0.0f);
    auto spec = frontend::logMelSpectrogram(silence.data(), silence.size());
    REQUIRE(spec.shape() == std::vector<size_t>{80, 3000});
    // log10(1e-10) = -10 for everything -> max-8 floor no-op -> (x+4)/4 = -1.5
    for (size_t i = 0; i < spec.size(); ++i) {
        REQUIRE(spec.data()[i] == Catch::Approx(-1.5).margin(1e-5));
    }
}

TEST_CASE("logmel: 440 Hz sine concentrates in right mel bins", "[frontend]") {
    // 1 second of 440 Hz sine at amplitude 0.5, then silence up to 30 s.
    std::vector<float> pcm(480000, 0.0f);
    for (size_t i = 0; i < 16000; ++i) {
        pcm[i] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) / 16000.0));
    }
    auto spec = frontend::logMelSpectrogram(pcm.data(), pcm.size());

    REQUIRE(spec.shape() == std::vector<size_t>{80, 3000});

    // average energy per mel bin over the first ~40 frames (covering the tone)
    std::vector<double> melEnergy(80, 0.0);
    for (size_t t = 0; t < 40; ++t) {
        for (size_t m = 0; m < 80; ++m) {
            melEnergy[m] += spec.data()[m + 80 * t];
        }
    }
    size_t argmax = 0;
    for (size_t m = 1; m < 80; ++m) {
        if (melEnergy[m] > melEnergy[argmax]) argmax = m;
    }
    // 440 Hz sits at mel 6.6 of 45.2 total -> mel index ~11-12
    INFO("argmax mel = " << argmax);
    REQUIRE(argmax >= 8);
    REQUIRE(argmax <= 16);

    // Whisper's normalization is global: silence frames are floored at
    // max-8 (in log10), so a strong tone lifts silent frames to (M-8+4)/4
    // = global_max_normalized - 2.
    float gmax = 0;
    for (size_t i = 0; i < spec.size(); ++i) gmax = std::max(gmax, spec.data()[i]);
    const float tail = spec.data()[40 + 80 * 2500];
    REQUIRE(tail == Catch::Approx(gmax - 2.0f).margin(1e-4));
}

TEST_CASE("logmel: 10 s input gives 1000 frames", "[frontend]") {
    std::vector<float> pcm(160000, 0.1f);
    auto spec = frontend::logMelSpectrogram(pcm.data(), pcm.size(), 160000);
    REQUIRE(spec.shape() == std::vector<size_t>{80, 1000});
}

TEST_CASE("logmel: window and filterbank match the tables embedded in the model", "[frontend][model]") {
    std::string path;
    if (!modelExists(path)) {
        WARN("no model file found, skipping embedded-table comparison");
        return;
    }

    gguf::File f = gguf::File::open(path);

    // hann window
    tensor::Tensor win = gguf::loadTensorF32(f, "frontend.window");
    REQUIRE(win.size() == 400);
    const auto ours = mel::hannWindow(400);
    double maxDiffWin = 0;
    for (size_t i = 0; i < 400; ++i) {
        maxDiffWin = std::max(maxDiffWin, std::abs(static_cast<double>(ours[i] - win.data()[i])));
    }
    INFO("window max diff = " << maxDiffWin);
    REQUIRE(maxDiffWin < 1e-5);

    // mel filterbank: embedded dims are {201, 80} (bin fastest), ours are [mel][bin]
    tensor::Tensor ref = gguf::loadTensorF32(f, "frontend.mel_filterbank");
    REQUIRE(ref.size() == 201 * 80);
    const auto fb = mel::filterbank(80, 400, 16000);
    double maxDiffFb = 0;
    for (size_t m = 0; m < 80; ++m) {
        for (size_t b = 0; b < 201; ++b) {
            maxDiffFb = std::max(maxDiffFb,
                                 std::abs(static_cast<double>(fb[m * 201 + b] - ref.data()[b + 201 * m])));
        }
    }
    INFO("filterbank max diff = " << maxDiffFb);
    REQUIRE(maxDiffFb < 1e-4);
}
