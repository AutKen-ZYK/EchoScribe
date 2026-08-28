#pragma once

#include <cstddef>
#include <vector>

namespace ops {

// Single-head scaled dot-product attention:
//   S = Q[Tq,dk] * K[Tk,dk]^T / sqrt(dk);  out = softmax(S + mask) * V[Tk,dv]
// All row-major. If causal, position j is only attended by query rows whose
// absolute index (i + qOffset) satisfies j <= i + qOffset.
void attention(const float* q, const float* k, const float* v, float* out, size_t tq, size_t tk,
               size_t dk, size_t dv, bool causal, size_t qOffset = 0);

// Multi-head attention over packed projections:
//   q: [tq, nHead*headDim], k/v: [tk, nHead*headDim], out: [tq, nHead*headDim]
// Head h uses columns [h*headDim, (h+1)*headDim). Heads are independent.
void multiHeadAttention(const float* q, const float* k, const float* v, float* out, size_t tq,
                        size_t tk, size_t nHead, size_t headDim, bool causal,
                        size_t qOffset = 0);

// Growable KV cache for autoregressive decoding.
// Stores per-layer keys/values of shape [maxTokens, nHead*headDim].
class KVCache {
public:
    KVCache() = default;
    KVCache(size_t nHead, size_t headDim, size_t maxTokens);

    void reset();

    // Appends T new key/value rows and computes attention of Tq query rows
    // (absolute positions len_..len_+Tq-1) over all len_+T cached positions.
    // q: [tq, nHead*headDim]; k/v: [t, nHead*headDim]; out: [tq, nHead*headDim].
    // Causal masking is always applied.
    void forward(const float* q, const float* k, const float* v, float* out, size_t tq, size_t t);

    size_t length() const { return len_; }

private:
    size_t nHead_ = 0;
    size_t headDim_ = 0;
    size_t len_ = 0;
    std::vector<float> k_;
    std::vector<float> v_;
};

} // namespace ops
