#include "ops/matmul.hpp"

#include "ops/simd.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ops {

void matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    // Row-at-a-time accumulation: C[i,:] = sum_k A[i,k] * B[k,:].
    // Tile i so B (K*N) stays hot in cache across the tile's rows.
    constexpr size_t kTile = 64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long iTile = 0; iTile < static_cast<long long>((m + kTile - 1) / kTile); ++iTile) {
        const size_t i0 = static_cast<size_t>(iTile) * kTile;
        const size_t iEnd = std::min(i0 + kTile, m);
        for (size_t i = i0; i < iEnd; ++i) {
            float* crow = c + i * n;
            std::memset(crow, 0, n * sizeof(float));
            const float* arow = a + i * k;
            for (size_t p = 0; p < k; ++p) {
                axpy(arow[p], b + p * n, crow, n);
            }
        }
    }
}

void linear(const float* x, const float* w, const float* bias, float* y, size_t m, size_t k,
            size_t n) {
    if (m <= 8) {
        // GEMV: y[m,n] = dot(x[m,:], w[n,:]) + bias[n]
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long nn = 0; nn < static_cast<long long>(n); ++nn) {
            const float* wr = w + static_cast<size_t>(nn) * k;
            const float b = bias ? bias[nn] : 0.0f;
            for (size_t i = 0; i < m; ++i) {
                y[i * n + nn] = dotProduct(x + i * k, wr, k) + b;
            }
        }
        return;
    }

    // Large M: compute y^T = W * X^T with an AXPY inner loop over m, then
    // transpose back. W rows and X^T rows are both k-contiguous; X^T stays
    // resident in cache while all N output rows stream over it.
    std::vector<float> xt(k * m);
    constexpr size_t kBlock = 32;
    for (size_t k0 = 0; k0 < k; k0 += kBlock) {
        const size_t kEnd = std::min(k0 + kBlock, k);
        for (size_t m0 = 0; m0 < m; m0 += kBlock) {
            const size_t mEnd = std::min(m0 + kBlock, m);
            for (size_t kk = k0; kk < kEnd; ++kk) {
                for (size_t i = m0; i < mEnd; ++i) {
                    xt[kk * m + i] = x[i * k + kk];
                }
            }
        }
    }

    std::vector<float> yt(n * m);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long nn = 0; nn < static_cast<long long>(n); ++nn) {
        const size_t nIdx = static_cast<size_t>(nn);
        float* yr = yt.data() + nIdx * m;
        const float b = bias ? bias[nIdx] : 0.0f;
        std::fill(yr, yr + m, b);
        const float* wr = w + nIdx * k;
        for (size_t kk = 0; kk < k; ++kk) {
            axpy(wr[kk], xt.data() + kk * m, yr, m);
        }
    }

    // y[m,n] = yt[n,m]
    for (size_t n0 = 0; n0 < n; n0 += kBlock) {
        const size_t nEnd = std::min(n0 + kBlock, n);
        for (size_t m0 = 0; m0 < m; m0 += kBlock) {
            const size_t mEnd = std::min(m0 + kBlock, m);
            for (size_t nn = n0; nn < nEnd; ++nn) {
                const float* src = yt.data() + nn * m;
                for (size_t i = m0; i < mEnd; ++i) {
                    y[i * n + nn] = src[i];
                }
            }
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
