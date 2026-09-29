#include "decoder/decoder.hpp"

#include "gguf/loader.hpp"
#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace whisper {

Decoder::Decoder(const gguf::File& f) {
    tokenEmbd_ = gguf::loadTensorF32(f, "dec.token_embd.weight");
    posEmb_ = gguf::loadTensorF32(f, "dec.pos_emb.weight");
    finalW_ = gguf::loadTensorF32(f, "dec.final_norm.weight");
    finalB_ = gguf::loadTensorF32(f, "dec.final_norm.bias");

    dModel_ = tokenEmbd_.shape()[1];
    nVocab_ = tokenEmbd_.shape()[0];
    nCtx_ = posEmb_.shape()[0];
    headDim_ = 64;
    nHead_ = dModel_ / headDim_;

    for (size_t i = 0;; ++i) {
        const std::string prefix = "dec.blocks." + std::to_string(i) + ".";
        if (!f.findTensor(prefix + "norm_self.weight")) break;
        Block b;
        b.normSelfW = gguf::loadTensorF32(f, prefix + "norm_self.weight");
        b.normSelfB = gguf::loadTensorF32(f, prefix + "norm_self.bias");
        b.qw = gguf::loadTensorF32(f, prefix + "self_attn.q.weight");
        b.qb = gguf::loadTensorF32(f, prefix + "self_attn.q.bias");
        b.kw = gguf::loadTensorF32(f, prefix + "self_attn.k.weight");
        b.vw = gguf::loadTensorF32(f, prefix + "self_attn.v.weight");
        b.vb = gguf::loadTensorF32(f, prefix + "self_attn.v.bias");
        b.ow = gguf::loadTensorF32(f, prefix + "self_attn.out.weight");
        b.ob = gguf::loadTensorF32(f, prefix + "self_attn.out.bias");
        b.normCrossW = gguf::loadTensorF32(f, prefix + "norm_cross.weight");
        b.normCrossB = gguf::loadTensorF32(f, prefix + "norm_cross.bias");
        b.cqW = gguf::loadTensorF32(f, prefix + "cross_attn.q.weight");
        b.cqB = gguf::loadTensorF32(f, prefix + "cross_attn.q.bias");
        b.ckW = gguf::loadTensorF32(f, prefix + "cross_attn.k.weight");
        b.cvW = gguf::loadTensorF32(f, prefix + "cross_attn.v.weight");
        b.cvB = gguf::loadTensorF32(f, prefix + "cross_attn.v.bias");
        b.coW = gguf::loadTensorF32(f, prefix + "cross_attn.out.weight");
        b.coB = gguf::loadTensorF32(f, prefix + "cross_attn.out.bias");
        b.normFfnW = gguf::loadTensorF32(f, prefix + "norm_ffn.weight");
        b.normFfnB = gguf::loadTensorF32(f, prefix + "norm_ffn.bias");
        b.fc1w = gguf::loadTensorF32(f, prefix + "ffn.fc1.weight");
        b.fc1b = gguf::loadTensorF32(f, prefix + "ffn.fc1.bias");
        b.fc2w = gguf::loadTensorF32(f, prefix + "ffn.fc2.weight");
        b.fc2b = gguf::loadTensorF32(f, prefix + "ffn.fc2.bias");
        blocks_.push_back(std::move(b));
    }
    if (blocks_.empty()) throw std::runtime_error("decoder: no blocks found in model");
}

void Decoder::forwardSequence(const tensor::Tensor& audio, const tensor::Tensor& x,
                              std::vector<ops::KVCache>& selfCaches,
                              const std::vector<tensor::Tensor>& crossK,
                              const std::vector<tensor::Tensor>& crossV,
                              tensor::Tensor& out) const {
    const size_t t = x.shape()[0];
    const size_t d = dModel_;
    const size_t nFfn = blocks_[0].fc1b.size();
    const size_t tAudio = audio.shape()[0];

    tensor::Tensor hn({t, d}), q({t, d}), k({t, d}), v({t, d}), att({t, d}), o({t, d}),
        f1({t, nFfn}), f2({t, d});
    tensor::Tensor cur = x; // [t, d]

    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        const Block& b = blocks_[bi];

        // x = x + self_attn(ln_self(x))   [causal, incremental cache]
        ops::layerNorm(cur.data(), b.normSelfW.data(), b.normSelfB.data(), hn.data(), t, d);
        ops::linear(hn.data(), b.qw.data(), b.qb.data(), q.data(), t, d, d);
        ops::linear(hn.data(), b.kw.data(), nullptr, k.data(), t, d, d);
        ops::linear(hn.data(), b.vw.data(), b.vb.data(), v.data(), t, d, d);
        selfCaches[bi].forward(q.data(), k.data(), v.data(), att.data(), t, t);
        ops::linear(att.data(), b.ow.data(), b.ob.data(), o.data(), t, d, d);
        ops::add(cur.data(), o.data(), cur.data(), cur.size());

        // x = x + cross_attn(ln_cross(x), audio)
        ops::layerNorm(cur.data(), b.normCrossW.data(), b.normCrossB.data(), hn.data(), t, d);
        ops::linear(hn.data(), b.cqW.data(), b.cqB.data(), q.data(), t, d, d);
        ops::multiHeadAttention(q.data(), crossK[bi].data(), crossV[bi].data(), att.data(), t,
                                tAudio, nHead_, headDim_, false);
        ops::linear(att.data(), b.coW.data(), b.coB.data(), o.data(), t, d, d);
        ops::add(cur.data(), o.data(), cur.data(), cur.size());

        // x = x + ffn(ln_ffn(x))
        ops::layerNorm(cur.data(), b.normFfnW.data(), b.normFfnB.data(), hn.data(), t, d);
        ops::linear(hn.data(), b.fc1w.data(), b.fc1b.data(), f1.data(), t, d, nFfn);
        ops::gelu(f1.data(), f1.data(), f1.size());
        ops::linear(f1.data(), b.fc2w.data(), b.fc2b.data(), f2.data(), t, nFfn, d);
        ops::add(cur.data(), f2.data(), cur.data(), cur.size());
    }

    ops::layerNorm(cur.data(), finalW_.data(), finalB_.data(), out.data(), t, d);
}

std::vector<size_t> Decoder::generate(const tensor::Tensor& audio, const Tokenizer& tok,
                                      const std::vector<size_t>& prompt, bool noTimestamps,
                                      size_t maxTokens, std::vector<float>* tokenLogprobs) const {
    const size_t d = dModel_;
    const size_t tAudio = audio.shape()[0];

    // Precompute cross-attention K/V from the encoder output once.
    std::vector<tensor::Tensor> crossK(blocks_.size()), crossV(blocks_.size());
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        const Block& b = blocks_[bi];
        crossK[bi] = tensor::Tensor({tAudio, d});
        crossV[bi] = tensor::Tensor({tAudio, d});
        ops::linear(audio.data(), b.ckW.data(), nullptr, crossK[bi].data(), tAudio, d, d);
        ops::linear(audio.data(), b.cvW.data(), b.cvB.data(), crossV[bi].data(), tAudio, d, d);
    }

    std::vector<ops::KVCache> selfCaches(blocks_.size());
    for (auto& c : selfCaches) c = ops::KVCache(nHead_, headDim_, nCtx_);

    std::vector<float> logits(nVocab());

    auto computeLogits = [&](const std::vector<size_t>& seq, size_t from, size_t count) {
        // Embed tokens[from .. from+count) at absolute positions from..; appends
        // to the self-attention caches; logits come from the last row.
        if (count == 0 || from + count > nCtx_) {
            throw std::runtime_error("decoder: token positions exceed the context window");
        }
        tensor::Tensor x({count, d});
        for (size_t i = 0; i < count; ++i) {
            const size_t tok = seq[from + i];
            const float* embd = tokenEmbd_.data() + tok * d;
            const float* pos = posEmb_.data() + (from + i) * d;
            float* row = x.data() + i * d;
            for (size_t j = 0; j < d; ++j) row[j] = embd[j] + pos[j];
        }
        tensor::Tensor hidden({count, d});
        forwardSequence(audio, x, selfCaches, crossK, crossV, hidden);
        // weight-tied projection: logits = hidden[last] @ embd^T
        ops::linear(hidden.data() + (count - 1) * d, tokenEmbd_.data(), nullptr, logits.data(), 1,
                    d, nVocab());
    };

    std::vector<size_t> seq = prompt;
    if (seq.empty() || seq.size() > nCtx_) {
        throw std::runtime_error("decoder: prompt of " + std::to_string(seq.size()) +
                                 " tokens does not fit the context window of " +
                                 std::to_string(nCtx_));
    }
    computeLogits(seq, 0, seq.size());

    const size_t tsBegin = tok.timestampBegin();

    std::vector<size_t> generated;
    for (size_t step = 0; step < maxTokens; ++step) {
        // One more token needs one more position: stop cleanly instead of
        // running past the positional embedding / KV cache and throwing
        // (whisper's own decoders cap the sample length the same way).
        if (seq.size() >= nCtx_) break;

        std::vector<float> masked = logits;
        for (int32_t id : tok.suppressTokens()) {
            if (id >= 0 && static_cast<size_t>(id) < nVocab()) {
                masked[static_cast<size_t>(id)] = -1e30f;
            }
        }
        if (step == 0) {
            for (int32_t id : tok.beginSuppressTokens()) {
                if (id >= 0 && static_cast<size_t>(id) < nVocab()) {
                    masked[static_cast<size_t>(id)] = -1e30f;
                }
            }
        }

        if (noTimestamps) {
            for (size_t id = tsBegin; id < nVocab(); ++id) {
                masked[id] = -1e30f;
            }
        } else {
            // whisper ApplyTimestampRules: <|notimestamps|> is prompt-only.
            masked[tok.notimestamps()] = -1e30f;

            const bool lastTs = !generated.empty() && generated.back() >= tsBegin;
            const bool penultTs = generated.size() < 2 || generated[generated.size() - 2] >= tsBegin;
            if (lastTs) {
                if (penultTs) {
                    // timestamp pair completed: next must be text
                    for (size_t id = tsBegin; id < nVocab(); ++id) masked[id] = -1e30f;
                } else {
                    // segment just ended: only eot or a later timestamp
                    for (size_t id = 0; id < tok.eot(); ++id) masked[id] = -1e30f;
                }
            }
            // timestamps must be strictly increasing within the sequence
            size_t lastTsId = 0;
            bool anyTs = false;
            for (size_t id : generated) {
                if (id >= tsBegin) {
                    lastTsId = id;
                    anyTs = true;
                }
            }
            if (anyTs) {
                const size_t tsLast = (lastTs && !penultTs) ? lastTsId : lastTsId + 1;
                for (size_t id = tsBegin; id < std::min(tsLast, nVocab()); ++id) {
                    masked[id] = -1e30f;
                }
            }
            if (generated.empty()) {
                // first generated token must be a timestamp within max_initial_timestamp (1 s)
                for (size_t id = 0; id < tsBegin; ++id) masked[id] = -1e30f;
                const size_t lastAllowed = tsBegin + 50; // 1.0 s / 0.02 s
                for (size_t id = lastAllowed + 1; id < nVocab(); ++id) masked[id] = -1e30f;
            }

            // if the summed timestamp logprob dominates all text tokens, force a timestamp
            double maxTs = -1e30;
            for (size_t id = tsBegin; id < nVocab(); ++id) {
                maxTs = std::max(maxTs, static_cast<double>(masked[id]));
            }
            double tsSum = 0.0;
            for (size_t id = tsBegin; id < nVocab(); ++id) {
                tsSum += std::exp(static_cast<double>(masked[id]) - maxTs);
            }
            const double tsLogProb = maxTs + std::log(tsSum);
            double maxText = -1e30;
            for (size_t id = 0; id < tsBegin; ++id) {
                maxText = std::max(maxText, static_cast<double>(masked[id]));
            }
            if (tsLogProb > maxText) {
                for (size_t id = 0; id < tsBegin; ++id) masked[id] = -1e30f;
            }
        }

        size_t next = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (size_t id = 0; id < nVocab(); ++id) {
            if (masked[id] > best) {
                best = masked[id];
                next = id;
            }
        }
        if (next == tok.eot()) break;

        if (tokenLogprobs) {
            // log-softmax of the suppressed logits over the full vocab
            double maxVal = -1e30;
            for (size_t id = 0; id < nVocab(); ++id) {
                maxVal = std::max(maxVal, static_cast<double>(masked[id]));
            }
            double sum = 0.0;
            for (size_t id = 0; id < nVocab(); ++id) {
                sum += std::exp(static_cast<double>(masked[id]) - maxVal);
            }
            tokenLogprobs->push_back(static_cast<float>(static_cast<double>(masked[next]) -
                                                        (maxVal + std::log(sum))));
        }

        seq.push_back(next);
        generated.push_back(next);
        computeLogits(seq, seq.size() - 1, 1);
    }
    return generated;
}

std::vector<float> Decoder::forwardLogits(const tensor::Tensor& audio,
                                          const std::vector<size_t>& tokens) const {
    const size_t d = dModel_;
    const size_t tAudio = audio.shape()[0];
    if (tokens.empty() || tokens.size() > nCtx_) {
        throw std::runtime_error("decoder: " + std::to_string(tokens.size()) +
                                 " tokens do not fit the context window of " +
                                 std::to_string(nCtx_));
    }

    std::vector<tensor::Tensor> crossK(blocks_.size()), crossV(blocks_.size());
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        const Block& b = blocks_[bi];
        crossK[bi] = tensor::Tensor({tAudio, d});
        crossV[bi] = tensor::Tensor({tAudio, d});
        ops::linear(audio.data(), b.ckW.data(), nullptr, crossK[bi].data(), tAudio, d, d);
        ops::linear(audio.data(), b.cvW.data(), b.cvB.data(), crossV[bi].data(), tAudio, d, d);
    }
    std::vector<ops::KVCache> selfCaches(blocks_.size());
    for (auto& c : selfCaches) c = ops::KVCache(nHead_, headDim_, nCtx_);

    tensor::Tensor x({tokens.size(), d});
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float* embd = tokenEmbd_.data() + tokens[i] * d;
        const float* pos = posEmb_.data() + i * d;
        float* row = x.data() + i * d;
        for (size_t j = 0; j < d; ++j) row[j] = embd[j] + pos[j];
    }
    tensor::Tensor hidden({tokens.size(), d});
    forwardSequence(audio, x, selfCaches, crossK, crossV, hidden);

    std::vector<float> logits(nVocab());
    ops::linear(hidden.data() + (tokens.size() - 1) * d, tokenEmbd_.data(), nullptr, logits.data(),
                1, d, nVocab());
    return logits;
}

} // namespace whisper
