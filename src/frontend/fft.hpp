#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace fft {

using Complex = std::complex<float>;

// In-place forward DFT (sign convention: X[k] = sum x[n] * exp(-2*pi*i*k*n/N)).
// Supports any N; factors N by 2 and 5, falls back to naive DFT for other primes.
void fft(std::vector<Complex>& a);

// Real-input DFT: returns bins 0..n/2 inclusive.
std::vector<Complex> fftReal(const float* x, size_t n);

} // namespace fft
