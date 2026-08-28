#include "ops/elementwise.hpp"

#include <cmath>

namespace ops {

void layerNorm(const float* x, const float* gamma, const float* beta, float* out, size_t rows,
               size_t n, float eps) {
    for (size_t i = 0; i < rows; ++i) {
        const float* row = x + i * n;
        float* orow = out + i * n;

        float mean = 0.0f;
        for (size_t j = 0; j < n; ++j) mean += row[j];
        mean /= static_cast<float>(n);

        float var = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            const float d = row[j] - mean;
            var += d * d;
        }
        var /= static_cast<float>(n);

        const float inv = 1.0f / std::sqrt(var + eps);
        for (size_t j = 0; j < n; ++j) {
            const float g = gamma ? gamma[j] : 1.0f;
            const float b = beta ? beta[j] : 0.0f;
            orow[j] = (row[j] - mean) * inv * g + b;
        }
    }
}

void gelu(const float* x, float* out, size_t n) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        const float inner = kSqrt2OverPi * (v + 0.044715f * v * v * v);
        out[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

void softmax(float* x, size_t nRows, size_t n) {
    for (size_t i = 0; i < nRows; ++i) {
        float* row = x + i * n;
        float maxVal = row[0];
        for (size_t j = 1; j < n; ++j) {
            maxVal = std::max(maxVal, row[j]);
        }
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            row[j] = std::exp(row[j] - maxVal);
            sum += row[j];
        }
        const float invSum = 1.0f / sum;
        for (size_t j = 0; j < n; ++j) {
            row[j] *= invSum;
        }
    }
}

} // namespace ops
