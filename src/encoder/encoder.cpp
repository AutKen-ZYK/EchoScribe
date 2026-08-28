#include "encoder/encoder.hpp"

#include "gguf/loader.hpp"
#include "ops/ops.hpp"

#include <stdexcept>

namespace whisper {

Encoder::Encoder(const gguf::File& f) {
    conv0w_ = gguf::loadTensorF32(f, "enc.conv.0.weight");
    conv0b_ = gguf::loadTensorF32(f, "enc.conv.0.bias");
    conv1w_ = gguf::loadTensorF32(f, "enc.conv.1.weight");
    conv1b_ = gguf::loadTensorF32(f, "enc.conv.1.bias");
    posEmb_ = gguf::loadTensorF32(f, "enc.pos_emb.weight");
    finalW_ = gguf::loadTensorF32(f, "enc.final_norm.weight");
    finalB_ = gguf::loadTensorF32(f, "enc.final_norm.bias");

    // Tensor shapes are torch order: conv weight [dModel, nMels, kernel]
    dModel_ = conv0w_.shape()[0];
    nHead_ = dModel_ / headDim_;

    // Count encoder blocks by probing names.
    for (size_t i = 0;; ++i) {
        const std::string prefix = "enc.blocks." + std::to_string(i) + ".";
        if (!f.findTensor(prefix + "norm_attn.weight")) break;
        Block b;
        b.normAttnW = gguf::loadTensorF32(f, prefix + "norm_attn.weight");
        b.normAttnB = gguf::loadTensorF32(f, prefix + "norm_attn.bias");
        b.qw = gguf::loadTensorF32(f, prefix + "attn.q.weight");
        b.qb = gguf::loadTensorF32(f, prefix + "attn.q.bias");
        b.kw = gguf::loadTensorF32(f, prefix + "attn.k.weight");
        b.vw = gguf::loadTensorF32(f, prefix + "attn.v.weight");
        b.vb = gguf::loadTensorF32(f, prefix + "attn.v.bias");
        b.ow = gguf::loadTensorF32(f, prefix + "attn.out.weight");
        b.ob = gguf::loadTensorF32(f, prefix + "attn.out.bias");
        b.normMlpW = gguf::loadTensorF32(f, prefix + "norm_ffn.weight");
        b.normMlpB = gguf::loadTensorF32(f, prefix + "norm_ffn.bias");
        b.fc1w = gguf::loadTensorF32(f, prefix + "ffn.fc1.weight");
        b.fc1b = gguf::loadTensorF32(f, prefix + "ffn.fc1.bias");
        b.fc2w = gguf::loadTensorF32(f, prefix + "ffn.fc2.weight");
        b.fc2b = gguf::loadTensorF32(f, prefix + "ffn.fc2.bias");
        blocks_.push_back(std::move(b));
    }
    if (blocks_.empty()) throw std::runtime_error("encoder: no blocks found in model");
}

tensor::Tensor Encoder::forward(const tensor::Tensor& mel) const {
    const size_t cIn = mel.shape()[0];
    const size_t tIn = mel.shape()[1];

    // conv0: [dModel, tIn], stride 1, padding 1
    tensor::Tensor h({dModel_, tIn});
    ops::conv1d(mel.data(), conv0w_.data(), conv0b_.data(), h.data(), cIn, dModel_, 3, tIn, 1, 1);
    ops::gelu(h.data(), h.data(), h.size());

    // conv1: [dModel, tIn/2], stride 2, padding 1
    const size_t tOut = tIn / 2;
    tensor::Tensor h2({dModel_, tOut});
    ops::conv1d(h.data(), conv1w_.data(), conv1b_.data(), h2.data(), dModel_, dModel_, 3, tIn, 1, 2);
    ops::gelu(h2.data(), h2.data(), h2.size());

    // channel-major [dModel, tOut] -> token-major [tOut, dModel]
    tensor::Tensor x({tOut, dModel_});
    for (size_t c = 0; c < dModel_; ++c) {
        for (size_t t = 0; t < tOut; ++t) {
            x.data()[t * dModel_ + c] = h2.data()[c * tOut + t];
        }
    }

    // add sinusoidal positional embedding (first tOut rows)
    ops::add(x.data(), posEmb_.data(), x.data(), x.size());

    const size_t d = dModel_;
    const size_t nFfn = blocks_[0].fc1b.size();
    std::vector<float> hn(tOut * d), q(tOut * d), k(tOut * d), v(tOut * d), att(tOut * d),
        o(tOut * d), f1(tOut * nFfn), f2(tOut * d);

    for (const Block& b : blocks_) {
        // x = x + attn(ln_attn(x))
        ops::layerNorm(x.data(), b.normAttnW.data(), b.normAttnB.data(), hn.data(), tOut, d);
        ops::linear(hn.data(), b.qw.data(), b.qb.data(), q.data(), tOut, d, d);
        ops::linear(hn.data(), b.kw.data(), nullptr, k.data(), tOut, d, d);
        ops::linear(hn.data(), b.vw.data(), b.vb.data(), v.data(), tOut, d, d);
        ops::multiHeadAttention(q.data(), k.data(), v.data(), att.data(), tOut, tOut, nHead_,
                                headDim_, false);
        ops::linear(att.data(), b.ow.data(), b.ob.data(), o.data(), tOut, d, d);
        ops::add(x.data(), o.data(), x.data(), x.size());

        // x = x + mlp(ln_mlp(x))
        ops::layerNorm(x.data(), b.normMlpW.data(), b.normMlpB.data(), hn.data(), tOut, d);
        ops::linear(hn.data(), b.fc1w.data(), b.fc1b.data(), f1.data(), tOut, d, nFfn);
        ops::gelu(f1.data(), f1.data(), f1.size());
        ops::linear(f1.data(), b.fc2w.data(), b.fc2b.data(), f2.data(), tOut, nFfn, d);
        ops::add(x.data(), f2.data(), x.data(), x.size());
    }

    tensor::Tensor out({tOut, d});
    ops::layerNorm(x.data(), finalW_.data(), finalB_.data(), out.data(), tOut, d);
    return out;
}

} // namespace whisper
