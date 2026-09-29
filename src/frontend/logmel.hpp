#pragma once

#include "tensor/tensor.hpp"

#include <cstddef>

namespace frontend {

// Whisper log-mel spectrogram frontend.
//
// Input: 16 kHz mono f32 samples (n <= targetSamples).
// Pipeline (identical to OpenAI whisper's log_mel_spectrogram):
//   1. zero-pad input to targetSamples
//   2. torch.stft semantics: reflect-pad 200 samples at both ends (center=True),
//      400-sample Hann window, 160-sample hop; drop the last frame
//   3. power spectrum (|X|^2, bins 0..200)
//   4. nMels-bin Slaney filterbank
//   5. log10(clamp 1e-10), floor at (max - 8), then (x + 4) / 4
//
// nMels must match the model: 80 for whisper tiny/base/small/medium and
// large-v1/v2, 128 for large-v3. Callers should use Encoder::nMels(), which
// reads it from the encoder conv weight, rather than assuming 80.
//
// Output: Tensor with shape {nMels, frames}, frames = targetSamples / 160.
// Linear layout matches torch's [nMels, frames] contiguous tensor.
// Throws if n > targetSamples, if nMels == 0, or if targetSamples < 160
// (which would yield zero frames).
tensor::Tensor logMelSpectrogram(const float* pcm, size_t n, size_t targetSamples = 480000,
                                 size_t nMels = 80);

} // namespace frontend
