#include "frontend/logmel.hpp"

#include "frontend/fft.hpp"
#include "frontend/mel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace frontend {
namespace {

constexpr size_t kSampleRate = 16000;
constexpr size_t kNfft = 400;
constexpr size_t kHop = 160;
constexpr size_t kNMels = 80;

} // namespace

tensor::Tensor logMelSpectrogram(const float* pcm, size_t n, size_t targetSamples) {
    if (n > targetSamples) {
        throw std::runtime_error("logmel: input longer than target length");
    }

    // Zero-pad to the target length.
    std::vector<float> padded(targetSamples, 0.0f);
    std::copy(pcm, pcm + n, padded.begin());

    // Reflect padding (torch.stft center=True semantics): left pad is
    // x[200], x[199], ..., x[1]; right pad mirrors the tail.
    std::vector<float> sig(targetSamples + kNfft, 0.0f);
    const size_t pad = kNfft / 2;
    for (size_t i = 0; i < pad; ++i) {
        sig[i] = padded[pad - i];       // x[pad], x[pad-1], ..., x[1]
        sig[targetSamples + pad + i] = padded[targetSamples - 2 - i];
    }
    std::copy(padded.begin(), padded.end(), sig.begin() + pad);

    const std::vector<float> window = mel::hannWindow(kNfft);
    const std::vector<float> filters = mel::filterbank(kNMels, kNfft, kSampleRate);
    constexpr size_t kNBins = kNfft / 2 + 1;

    const size_t frames = targetSamples / kHop;
    tensor::Tensor spec({kNMels, frames});
    float* out = spec.data();

    std::vector<float> frame(kNfft);
    std::vector<float> power(kNBins);
    std::vector<float> mel(kNMels);
    std::vector<fft::Complex> spectrum;

    for (size_t t = 0; t < frames; ++t) {
        const float* seg = sig.data() + t * kHop;
        for (size_t i = 0; i < kNfft; ++i) {
            frame[i] = seg[i] * window[i];
        }

        spectrum = fft::fftReal(frame.data(), kNfft);
        for (size_t b = 0; b < kNBins; ++b) {
            power[b] = std::norm(spectrum[b]); // |X|^2
        }

        for (size_t m = 0; m < kNMels; ++m) {
            float acc = 0.0f;
            const float* row = filters.data() + m * kNBins;
            for (size_t b = 0; b < kNBins; ++b) {
                acc += row[b] * power[b];
            }
            mel[m] = acc;
        }

        // log10 with clamp, then normalize against the global max.
        for (size_t m = 0; m < kNMels; ++m) {
            const float v = std::max(mel[m], 1e-10f);
            out[m + kNMels * t] = std::log10(v);
        }
    }

    // Global max across all frames and mels, floor at max - 8, rescale.
    float* data = spec.data();
    const float maxVal = *std::max_element(data, data + spec.size());
    const float floorVal = maxVal - 8.0f;
    for (size_t i = 0; i < spec.size(); ++i) {
        const float v = std::max(data[i], floorVal);
        data[i] = (v + 4.0f) / 4.0f;
    }
    return spec;
}

} // namespace frontend
