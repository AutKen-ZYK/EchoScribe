#include "ops/conv1d.hpp"

#include "ops/elementwise.hpp"
#include "ops/simd.hpp"

#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ops {

void conv1d(const float* x, const float* w, const float* bias, float* y, size_t cIn, size_t cOut,
            size_t kernel, size_t t, size_t padding, size_t stride) {
    const long long T = static_cast<long long>(t);
    const size_t tOut = (t + 2 * padding - kernel) / stride + 1;

    // Accumulation formulation: y[o,:] = bias[o] + sum_{c,k} w[o,c,k] * x[c, ti*s + k - p]
    // The inner loop over ti is contiguous for stride == 1 (axpy-able).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long oo = 0; oo < static_cast<long long>(cOut); ++oo) {
        const size_t o = static_cast<size_t>(oo);
        float* orow = y + o * tOut;
        const float b = bias ? bias[o] : 0.0f;
        std::fill(orow, orow + tOut, b);

        for (size_t c = 0; c < cIn; ++c) {
            const float* xrow = x + c * t;
            const float* wrow = w + o * (cIn * kernel) + c * kernel;
            for (size_t k = 0; k < kernel; ++k) {
                const long long offset = static_cast<long long>(k) - static_cast<long long>(padding);
                // valid ti: 0 <= ti*stride + offset < T
                long long ti0 = 0;
                if (offset < 0) ti0 = (-offset + static_cast<long long>(stride) - 1) / static_cast<long long>(stride);
                long long ti1 = tOut;
                if (offset >= 0) {
                    ti1 = std::min<long long>(tOut, (T - 1 - offset) / static_cast<long long>(stride) + 1);
                }
                const float wk = wrow[k];
                if (stride == 1) {
                    const float* xsrc = xrow + offset;
                    for (long long ti = ti0; ti < ti1; ++ti) {
                        orow[ti] += wk * xsrc[ti];
                    }
                } else {
                    for (long long ti = ti0; ti < ti1; ++ti) {
                        orow[ti] += wk * xrow[static_cast<size_t>(ti) * stride + offset];
                    }
                }
            }
        }
    }
}

void gelu2d(const float* x, float* out, size_t c, size_t t) {
    gelu(x, out, c * t);
}

} // namespace ops
