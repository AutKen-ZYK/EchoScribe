#include "frontend/fft.hpp"

#include <cmath>

namespace fft {
namespace {

constexpr double kPi = 3.14159265358979323846;

Complex twiddle(size_t k, size_t n) {
    const double a = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(n);
    return Complex(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
}

// e^(-2*pi*i*j/5) for j = 0..4, exact constants to double precision.
const Complex kTwiddle5[5] = {
    {1.0f, 0.0f},
    {0.30901699437494745f, -0.9510565162951535f},
    {-0.8090169943749473f, -0.5877852522924732f},
    {-0.8090169943749473f, 0.5877852522924732f},
    {0.30901699437494745f, 0.9510565162951535f},
};

// out[k] for k = 0..n-1 gets the DFT of x[0], x[stride], x[2*stride], ...
void fftRec(const Complex* x, size_t stride, size_t n, Complex* out) {
    if (n == 1) {
        out[0] = x[0];
        return;
    }

    if (n % 2 == 0) {
        const size_t h = n / 2;
        std::vector<Complex> e(h), o(h);
        fftRec(x, stride * 2, h, e.data());
        fftRec(x + stride, stride * 2, h, o.data());
        for (size_t k = 0; k < h; ++k) {
            const Complex t = twiddle(k, n) * o[k];
            out[k] = e[k] + t;
            out[k + h] = e[k] - t;
        }
        return;
    }

    if (n % 5 == 0) {
        const size_t m = n / 5;
        std::vector<Complex> f[5];
        for (size_t j = 0; j < 5; ++j) {
            f[j].resize(m);
            fftRec(x + j * stride, stride * 5, m, f[j].data());
        }
        // X[q + s*m] = sum_j w5^(s*j) * w_n^(q*j) * F_j[q]
        for (size_t q = 0; q < m; ++q) {
            Complex t[5];
            for (size_t j = 0; j < 5; ++j) {
                t[j] = f[j][q] * twiddle(q * j, n);
            }
            for (size_t s = 0; s < 5; ++s) {
                Complex acc{0.0f, 0.0f};
                for (size_t j = 0; j < 5; ++j) {
                    acc += t[j] * kTwiddle5[(s * j) % 5];
                }
                out[q + s * m] = acc;
            }
        }
        return;
    }

    // Other prime factors (3, 7, ...): naive DFT.
    for (size_t s = 0; s < n; ++s) {
        Complex acc{0.0f, 0.0f};
        for (size_t j = 0; j < n; ++j) {
            acc += x[j * stride] * twiddle((s * j) % n, n);
        }
        out[s] = acc;
    }
}

} // namespace

void fft(std::vector<Complex>& a) {
    const size_t n = a.size();
    if (n <= 1) return;
    std::vector<Complex> out(n);
    fftRec(a.data(), 1, n, out.data());
    a.swap(out);
}

std::vector<Complex> fftReal(const float* x, size_t n) {
    std::vector<Complex> a(n);
    for (size_t i = 0; i < n; ++i) a[i] = Complex(x[i], 0.0f);
    fft(a);
    a.resize(n / 2 + 1);
    return a;
}

} // namespace fft
