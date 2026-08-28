#include "ops/attention.hpp"

#include "ops/elementwise.hpp"
#include "ops/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ops {

void attention(const float* q, const float* k, const float* v, float* out, size_t tq, size_t tk,
               size_t dk, size_t dv, bool causal, size_t qOffset) {
    // scores[i, j] = q[i] . k[j] / sqrt(dk)
    std::vector<float> scores(tq * tk);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(tq); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        const float* qrow = q + i * dk;
        for (size_t j = 0; j < tk; ++j) {
            scores[i * tk + j] = ops::dotProduct(qrow, k + j * dk, dk) * scale;
        }
        if (causal) {
            const size_t limit = i + qOffset;
            for (size_t j = limit + 1; j < tk; ++j) {
                scores[i * tk + j] = -1e30f; // excluded after softmax
            }
        }
    }
    softmax(scores.data(), tq, tk);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(tq); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        float* orow = out + i * dv;
        std::memset(orow, 0, dv * sizeof(float));
        const float* prow = scores.data() + i * tk;
        for (size_t j = 0; j < tk; ++j) {
            ops::axpy(prow[j], v + j * dv, orow, dv);
        }
    }
}

void multiHeadAttention(const float* q, const float* k, const float* v, float* out, size_t tq,
                        size_t tk, size_t nHead, size_t headDim, bool causal, size_t qOffset) {
    const size_t dModel = nHead * headDim;
    std::vector<float> qh(tq * headDim), kh(tk * headDim), vh(tk * headDim),
        oh(tq * headDim);

    for (size_t h = 0; h < nHead; ++h) {
        // extract head h (strided copy) into contiguous buffers
        for (size_t i = 0; i < tq; ++i) {
            std::memcpy(qh.data() + i * headDim, q + i * dModel + h * headDim,
                        headDim * sizeof(float));
        }
        for (size_t j = 0; j < tk; ++j) {
            std::memcpy(kh.data() + j * headDim, k + j * dModel + h * headDim,
                        headDim * sizeof(float));
            std::memcpy(vh.data() + j * headDim, v + j * dModel + h * headDim,
                        headDim * sizeof(float));
        }

        attention(qh.data(), kh.data(), vh.data(), oh.data(), tq, tk, headDim, headDim, causal,
                  qOffset);

        for (size_t i = 0; i < tq; ++i) {
            std::memcpy(out + i * dModel + h * headDim, oh.data() + i * headDim,
                        headDim * sizeof(float));
        }
    }
}

KVCache::KVCache(size_t nHead, size_t headDim, size_t maxTokens)
    : nHead_(nHead),
      headDim_(headDim),
      k_(maxTokens * nHead * headDim, 0.0f),
      v_(maxTokens * nHead * headDim, 0.0f) {}

void KVCache::reset() {
    len_ = 0;
}

void KVCache::forward(const float* q, const float* k, const float* v, float* out, size_t tq,
                      size_t t) {
    const size_t dModel = nHead_ * headDim_;
    if (len_ + t > k_.size() / dModel) {
        throw std::runtime_error("kvcache: overflow");
    }
    std::memcpy(k_.data() + len_ * dModel, k, t * dModel * sizeof(float));
    std::memcpy(v_.data() + len_ * dModel, v, t * dModel * sizeof(float));
    multiHeadAttention(q, k_.data(), v_.data(), out, tq, len_ + t, nHead_, headDim_, true, len_);
    len_ += t;
}

} // namespace ops
