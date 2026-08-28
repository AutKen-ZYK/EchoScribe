#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "frontend/fft.hpp"

#include <cmath>
#include <complex>
#include <vector>

using fft::Complex;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

namespace {

// O(n^2) reference DFT, computed in double precision.
std::vector<Complex> naiveDft(const std::vector<Complex>& x) {
    const size_t n = x.size();
    std::vector<Complex> out(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (size_t t = 0; t < n; ++t) {
            const double a = -2.0 * M_PI * static_cast<double>(k) * static_cast<double>(t) /
                             static_cast<double>(n);
            acc += std::complex<double>(x[t]) * std::complex<double>(std::cos(a), std::sin(a));
        }
        out[k] = Complex(static_cast<float>(acc.real()), static_cast<float>(acc.imag()));
    }
    return out;
}

double maxDiff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(std::complex<double>(a[i] - b[i])));
    }
    return d;
}

} // namespace

TEST_CASE("fft matches naive dft on many sizes", "[fft]") {
    for (size_t n : {size_t{2}, size_t{3}, size_t{4}, size_t{5}, size_t{7}, size_t{8}, size_t{16},
                     size_t{25}, size_t{32}, size_t{40}, size_t{100}, size_t{125}, size_t{200},
                     size_t{400}}) {
        std::vector<Complex> x(n);
        for (size_t i = 0; i < n; ++i) {
            const double a = 2.0 * M_PI * static_cast<double>(i) * 0.7 +
                             0.3 * static_cast<double>(i * i % 7);
            x[i] = Complex(static_cast<float>(std::cos(a) + 0.5 * std::sin(2.3 * a)),
                           static_cast<float>(std::sin(a) * 0.8));
        }
        auto ref = naiveDft(x);
        fft::fft(x);
        // f32 accumulation over n^2 terms for the reference: loose relative bound.
        const double scale = 2.0 * static_cast<double>(n);
        INFO("n=" << n);
        REQUIRE(maxDiff(x, ref) < 1e-3 * scale / 8.0);
    }
}

TEST_CASE("fft impulse gives flat spectrum", "[fft]") {
    std::vector<Complex> x(8, {0.0f, 0.0f});
    x[0] = {1.0f, 0.0f};
    fft::fft(x);
    for (const auto& v : x) {
        REQUIRE_THAT(v.real(), WithinAbs(1.0, 1e-5));
        REQUIRE_THAT(v.imag(), WithinAbs(0.0, 1e-5));
    }
}

TEST_CASE("fft single tone gives single bin", "[fft]") {
    // x[k] = exp(2*pi*i*2*k/16) -> X[2] = 16, all other bins 0.
    std::vector<Complex> x(16);
    for (size_t k = 0; k < 16; ++k) {
        const double a = 2.0 * M_PI * 2.0 * static_cast<double>(k) / 16.0;
        x[k] = Complex(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
    }
    fft::fft(x);
    for (size_t k = 0; k < 16; ++k) {
        const double expect = (k == 2) ? 16.0 : 0.0;
        REQUIRE(std::abs(std::complex<double>(x[k])) == Catch::Approx(expect).margin(1e-3));
    }
}

TEST_CASE("fft real sine peaks at correct bin with correct amplitude", "[fft]") {
    // 400-point window, sine at bin 40 with amplitude 1: |X[40]| = N/2 = 200.
    constexpr size_t n = 400;
    std::vector<float> s(n);
    for (size_t k = 0; k < n; ++k) {
        s[k] = static_cast<float>(std::sin(2.0 * M_PI * 40.0 * static_cast<double>(k) / 400.0));
    }
    auto X = fft::fftReal(s.data(), n);
    REQUIRE(X.size() == n / 2 + 1);

    double best = 0;
    size_t argmax = 0;
    for (size_t b = 0; b < X.size(); ++b) {
        const double m = std::abs(std::complex<double>(X[b]));
        if (m > best) {
            best = m;
            argmax = b;
        }
    }
    REQUIRE(argmax == 40);
    REQUIRE(best == Catch::Approx(200.0).margin(0.05));
}

TEST_CASE("fft satisfies parseval", "[fft]") {
    constexpr size_t n = 128;
    std::vector<Complex> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = Complex(static_cast<float>(std::sin(0.123 * static_cast<double>(i))),
                       static_cast<float>(std::cos(0.456 * static_cast<double>(i))));
    }
    double energyTime = 0;
    for (const auto& v : x) energyTime += std::norm(v);
    fft::fft(x);
    double energyFreq = 0;
    for (const auto& v : x) energyFreq += std::norm(v);
    REQUIRE(energyFreq / static_cast<double>(n) ==
            Catch::Approx(energyTime).margin(1e-2 * energyTime));
}
