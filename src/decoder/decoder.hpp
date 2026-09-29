#pragma once

#include "gguf/gguf.hpp"
#include "ops/attention.hpp"
#include "tensor/tensor.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstddef>
#include <vector>

namespace whisper {

// Whisper text decoder with weight-tied output projection (logits = x @ embd^T),
// incremental self-attention KV cache and precomputed cross-attention K/V.
class Decoder {
public:
    explicit Decoder(const gguf::File& f);

    // Greedy decoding. prompt is the initial token sequence (sot, language,
    // transcribe[, notimestamps]); generated tokens (excluding the prompt) are
    // returned. Stops at eot, after maxTokens generated tokens, or when the
    // context window (nCtx) is full - it never generates past the positional
    // embedding. If tokenLogprobs is non-null it receives the per-token
    // log-probability (after suppression rules) of each generated token.
    std::vector<size_t> generate(const tensor::Tensor& audio, const Tokenizer& tok,
                                 const std::vector<size_t>& prompt, bool noTimestamps,
                                 size_t maxTokens, std::vector<float>* tokenLogprobs = nullptr) const;

    // Logits for a fixed token sequence (no caching); useful for tests.
    std::vector<float> forwardLogits(const tensor::Tensor& audio,
                                     const std::vector<size_t>& tokens) const;

    size_t dModel() const { return dModel_; }
    size_t nHead() const { return nHead_; }
    size_t nLayers() const { return blocks_.size(); }
    size_t nVocab() const { return nVocab_; }
    size_t nCtx() const { return nCtx_; }

private:
    struct Block {
        tensor::Tensor normSelfW, normSelfB;
        tensor::Tensor qw, qb, kw, vw, vb, ow, ob;
        tensor::Tensor normCrossW, normCrossB;
        tensor::Tensor cqW, cqB, ckW, cvW, cvB, coW, coB;
        tensor::Tensor normFfnW, normFfnB;
        tensor::Tensor fc1w, fc1b, fc2w, fc2b;
    };

    // x: [t, dModel] embeddings; returns LN-final hidden states [t, dModel].
    void forwardSequence(const tensor::Tensor& audio, const tensor::Tensor& x,
                         std::vector<ops::KVCache>& selfCaches,
                         const std::vector<tensor::Tensor>& crossK,
                         const std::vector<tensor::Tensor>& crossV,
                         tensor::Tensor& out) const;

    size_t dModel_ = 0;
    size_t nHead_ = 0;
    size_t headDim_ = 64;
    size_t nCtx_ = 448;
    size_t nVocab_ = 0;
    tensor::Tensor tokenEmbd_; // [nVocab, dModel]
    tensor::Tensor posEmb_;    // [nCtx, dModel]
    tensor::Tensor finalW_, finalB_;
    std::vector<Block> blocks_;
};

} // namespace whisper
