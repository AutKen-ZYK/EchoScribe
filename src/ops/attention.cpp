#include "ops/attention.hpp"

#include "ops/elementwise.hpp"
#include "ops/matmul.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ops {

void attention(const float* q, const float* k, const float* v, float* out, size_t tq, size_t tk,
               size_t dk, size_t dv, bool causal, size_t qOffset) {
    // scores[i, j] = q[i] . k[j] / sqrt(dk)
    std::vector<float> scores(tq * tk);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
    for (size_t i = 0; i < tq; ++i) {
        for (size_t j = 0; j < tk; ++j) {
            float acc = 0.0f;
            for (size_t d = 0; d < dk; ++d) {
                acc += q[i * dk + d] * k[j * dk + d];
            }
            scores[i * tk + j] = acc * scale;
        }
        if (causal) {
            const size_t limit = i + qOffset;
            for (size_t j = limit + 1; j < tk; ++j) {
                scores[i * tk + j] = -1e30f; // excluded after softmax
            }
        }
    }
    softmax(scores.data(), tq, tk);

    for (size_t i = 0; i < tq; ++i) {
        float* orow = out + i * dv;
        std::memset(orow, 0, dv * sizeof(float));
        for (size_t j = 0; j < tk; ++j) {
            const float p = scores[i * tk + j];
            const float* vrow = v + j * dv;
            for (size_t d = 0; d < dv; ++d) {
                orow[d] += p * vrow[d];
            }
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
