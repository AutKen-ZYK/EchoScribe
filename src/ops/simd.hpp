#pragma once

#include <cstddef>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// Portable SIMD dot-product and axpy kernels used by the operator library.
// AVX2+FMA path when compiled with -mavx2 -mfma, scalar fallback otherwise.

namespace ops {

inline float dotProduct(const float* a, const float* b, size_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    float tmp[4];
    _mm_storeu_ps(tmp, lo);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

// out[0..m) += alpha * src[0..m)
inline void axpy(float alpha, const float* src, float* out, size_t m) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 va = _mm256_set1_ps(alpha);
    size_t i = 0;
    for (; i + 8 <= m; i += 8) {
        const __m256 vv = _mm256_loadu_ps(src + i);
        __m256 vo = _mm256_loadu_ps(out + i);
        vo = _mm256_fmadd_ps(va, vv, vo);
        _mm256_storeu_ps(out + i, vo);
    }
    for (; i < m; ++i) out[i] += alpha * src[i];
#else
    for (size_t i = 0; i < m; ++i) out[i] += alpha * src[i];
#endif
}

} // namespace ops
