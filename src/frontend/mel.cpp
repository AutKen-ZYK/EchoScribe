#include "frontend/mel.hpp"

#include <cmath>

namespace mel {
namespace {

constexpr double kLn6_4 = 1.8562979903656327; // ln(6.4)
constexpr double kPi = 3.14159265358979323846;

// Slaney auditory scale (Malcolm Slaney, Auditory Toolbox), used by librosa
// with htk=False: mel = 3f/200 below 1 kHz, log above.
double fToMel(double f) {
    if (f < 1000.0) return 3.0 * f / 200.0;
    return 15.0 + 27.0 * std::log(f / 1000.0) / kLn6_4;
}

double melToF(double m) {
    if (m < 15.0) return 200.0 * m / 3.0;
    return 1000.0 * std::exp(kLn6_4 * (m - 15.0) / 27.0);
}

} // namespace

std::vector<float> hannWindow(size_t n) {
    std::vector<float> w(n);
    for (size_t k = 0; k < n; ++k) {
        w[k] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(k) / static_cast<double>(n)));
    }
    return w;
}

std::vector<float> filterbank(size_t nMels, size_t nFft, size_t sampleRate) {
    const size_t nBins = nFft / 2 + 1;
    std::vector<float> w(nMels * nBins, 0.0f);

    // FFT bin center frequencies: sr * k / nFft, k = 0..nBins-1
    std::vector<double> fftFreq(nBins);
    for (size_t k = 0; k < nBins; ++k) {
        fftFreq[k] = static_cast<double>(sampleRate * k) / static_cast<double>(nFft);
    }

    // nMels+2 triangle boundary frequencies, spaced linearly in mel between fmin and fmax.
    const double fmin = 0.0;
    const double fmax = static_cast<double>(sampleRate) / 2.0;
    const double melMin = fToMel(fmin);
    const double melMax = fToMel(fmax);
    std::vector<double> melPts(nMels + 2), fPts(nMels + 2);
    for (size_t i = 0; i < nMels + 2; ++i) {
        melPts[i] = melMin + (melMax - melMin) * static_cast<double>(i) / static_cast<double>(nMels + 1);
        fPts[i] = melToF(melPts[i]);
    }

    // Slaney normalization: each triangle is scaled by 2 / (width of its base).
    // NOTE: mel_f here are Hz values (mel_frequencies returns Hz), so the base
    // width is measured in Hz, not in mel units.
    std::vector<double> enorm(nMels);
    for (size_t i = 0; i < nMels; ++i) {
        enorm[i] = 2.0 / (fPts[i + 2] - fPts[i]);
    }

    for (size_t m = 0; m < nMels; ++m) {
        const double lower = fPts[m];
        const double center = fPts[m + 1];
        const double upper = fPts[m + 2];
        for (size_t k = 0; k < nBins; ++k) {
            const double f = fftFreq[k];
            double v = 0.0;
            if (lower <= f && f <= center && center > lower) {
                v = (f - lower) / (center - lower);
            } else if (center < f && f <= upper && upper > center) {
                v = (upper - f) / (upper - center);
            }
            w[m * nBins + k] = static_cast<float>(v * enorm[m]);
        }
    }
    return w;
}

} // namespace mel
