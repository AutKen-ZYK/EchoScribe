#pragma once

#include "gguf/gguf.hpp"
#include "tensor/tensor.hpp"

#include <cstddef>
#include <vector>

namespace whisper {

// Whisper audio encoder, loaded from a GGUF file (llama.cpp-style names:
// enc.conv.*, enc.blocks.N.*, enc.pos_emb.weight, enc.final_norm.*).
class Encoder {
public:
    explicit Encoder(const gguf::File& f);

    // mel: channel-major [nMels, frames] (log-mel frontend output), frames even.
    // The channel count must equal nMels(); the frontend should be built with
    // the same value or the first conv reads the wrong weight rows.
    // Returns token-major [frames/2, dModel], LayerNorm applied at the end.
    tensor::Tensor forward(const tensor::Tensor& mel) const;

    size_t dModel() const { return dModel_; }
    size_t nHead() const { return nHead_; }
    size_t nLayers() const { return blocks_.size(); }
    // Mel bins expected by the first conv, from its weight shape: 80 for whisper
    // tiny..large-v2, 128 for large-v3.
    size_t nMels() const { return nMels_; }

private:
    struct Block {
        tensor::Tensor normAttnW, normAttnB;
        tensor::Tensor qw, qb, kw, vw, vb, ow, ob; // no k bias in whisper
        tensor::Tensor normMlpW, normMlpB;
        tensor::Tensor fc1w, fc1b, fc2w, fc2b;
    };

    size_t dModel_ = 0;
    size_t nHead_ = 0;
    size_t nMels_ = 0;
    size_t headDim_ = 64;
    tensor::Tensor conv0w_, conv0b_, conv1w_, conv1b_;
    tensor::Tensor posEmb_;
    tensor::Tensor finalW_, finalB_;
    std::vector<Block> blocks_;
};

} // namespace whisper
