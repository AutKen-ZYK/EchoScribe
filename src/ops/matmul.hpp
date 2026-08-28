#pragma once

#include <cstddef>

namespace ops {

// C[M,N] = A[M,K] * B[K,N], all row-major (last dim contiguous).
void matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n);

// y[M,N] = x[M,K] * W[N,K]^T + b, i.e. torch.nn.Linear semantics.
// W is [N,K] row-major, b is [N] (may be null).
void linear(const float* x, const float* w, const float* bias, float* y, size_t m, size_t k,
            size_t n);

// x[M,N] += b (broadcast over rows).
void addBias(float* x, const float* b, size_t m, size_t n);

// out = a + b (elementwise, n elements).
void add(const float* a, const float* b, float* out, size_t n);

// out = a + b * scale (elementwise).
void addScaled(const float* a, const float* b, float scale, float* out, size_t n);

} // namespace ops
