#pragma once

#include <cstddef>

namespace ops {

// Row-wise LayerNorm over rows of length n:
// out[i,j] = (x[i,j] - mean_i) / sqrt(var_i + eps) * gamma[j] + beta[j].
// Variance is biased (1/n), matching torch.nn.LayerNorm. gamma/beta may be null (=1/0).
void layerNorm(const float* x, const float* gamma, const float* beta, float* out, size_t rows,
               size_t n, float eps = 1e-5f);

// GELU with tanh approximation, as used by Whisper:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(const float* x, float* out, size_t n);

// Row-wise, numerically stable softmax (in-place), nRows rows of length n.
void softmax(float* x, size_t nRows, size_t n);

} // namespace ops
