#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace audio {

struct WavData {
    std::vector<float> samples; // mono, f32 in [-1, 1]
    size_t sampleRate = 0;
};

// Minimal WAV reader: PCM16 / PCM32 / float32, multi-channel (downmixed by
// averaging), linear-interpolation resample to a target rate if needed.
WavData readWav(const std::string& path, size_t targetRate = 16000);

} // namespace audio
