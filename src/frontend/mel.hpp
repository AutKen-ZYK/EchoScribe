#pragma once

#include <cstddef>
#include <vector>

namespace mel {

// Periodic Hann window, identical to torch.hann_window(n) / librosa 'hann' fftbins:
// w[k] = 0.5 - 0.5 * cos(2*pi*k/n), k = 0..n-1.
std::vector<float> hannWindow(size_t n);

// Slaney-scale, Slaney-normalized mel filterbank, matching
// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm="slaney"),
// which is what OpenAI Whisper uses to build mel_filters.npz.
// Row-major [nMels][nBins], nBins = nFft/2 + 1, linear index = mel * nBins + bin.
std::vector<float> filterbank(size_t nMels, size_t nFft, size_t sampleRate);

} // namespace mel
