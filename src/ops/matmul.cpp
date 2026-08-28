#include "ops/matmul.hpp"

#include <cstring>

namespace ops {

void matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    // i-k-j order: streams B rows and keeps C row hot in cache.
    for (size_t i = 0; i < m; ++i) {
        float* crow = c + i * n;
        std::memset(crow, 0, n * sizeof(float));
        const float* arow = a + i * k;
        for (size_t p = 0; p < k; ++p) {
            const float av = arow[p];
            const float* brow = b + p * n;
            for (size_t j = 0; j < n; ++j) {
                crow[j] += av * brow[j];
            }
        }
    }
}

void linear(const float* x, const float* w, const float* bias, float* y, size_t m, size_t k,
            size_t n) {
    for (size_t i = 0; i < m; ++i) {
        const float* xrow = x + i * k;
        float* yrow = y + i * n;
        for (size_t j = 0; j < n; ++j) {
            const float* wrow = w + j * k;
            float acc = bias ? bias[j] : 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += xrow[p] * wrow[p];
            }
            yrow[j] = acc;
        }
    }
}

void addBias(float* x, const float* b, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        float* row = x + i * n;
        for (size_t j = 0; j < n; ++j) {
            row[j] += b[j];
        }
    }
}

void add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void addScaled(const float* a, const float* b, float scale, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i] * scale;
}

} // namespace ops
