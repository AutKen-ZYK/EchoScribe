#pragma once

#include <cstddef>

namespace ops {

// 1D convolution, torch.nn.Conv1d semantics.
//
// x: [cIn, t] channel-major (torch [B, C, T] contiguous without batch dim).
// w: [cOut, cIn, kernel] torch layout, linear index = o*(cIn*K) + c*K + k.
// bias: [cOut], may be null. padding: zero padding on both sides (whisper uses 1).
// stride: output stride (whisper's second conv uses 2).
// y: [cOut, tOut] where tOut = (t + 2*padding - kernel) / stride + 1.
void conv1d(const float* x, const float* w, const float* bias, float* y, size_t cIn,
            size_t cOut, size_t kernel, size_t t, size_t padding, size_t stride = 1);

// GELU applied channel-wise over [c, t] (in-place allowed via separate output).
void gelu2d(const float* x, float* out, size_t c, size_t t);

} // namespace ops
