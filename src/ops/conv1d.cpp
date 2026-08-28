#include "ops/conv1d.hpp"

#include "ops/elementwise.hpp"

namespace ops {

void conv1d(const float* x, const float* w, const float* bias, float* y, size_t cIn, size_t cOut,
            size_t kernel, size_t t, size_t padding) {
    const long long T = static_cast<long long>(t);
    for (size_t o = 0; o < cOut; ++o) {
        float* orow = y + o * t;
        for (long long ti = 0; ti < T; ++ti) {
            float acc = bias ? bias[o] : 0.0f;
            for (size_t c = 0; c < cIn; ++c) {
                const float* xrow = x + c * t;
                const float* wrow = w + o * (cIn * kernel) + c * kernel;
                for (size_t k = 0; k < kernel; ++k) {
                    const long long src = ti + static_cast<long long>(k) - static_cast<long long>(padding);
                    if (src >= 0 && src < T) {
                        acc += xrow[src] * wrow[k];
                    }
                }
            }
            orow[ti] = acc;
        }
    }
}

void gelu2d(const float* x, float* out, size_t c, size_t t) {
    gelu(x, out, c * t);
}

} // namespace ops
