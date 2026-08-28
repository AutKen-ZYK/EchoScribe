#pragma once

#include "streaming/stream.hpp"

#include <string>

namespace audio {

// Microphone capture via miniaudio (WASAPI/ALSA/PulseAudio auto-selected).
// The device callback pushes 16 kHz mono f32 frames into the ring buffer.
class MicCapture {
public:
    MicCapture() = default;
    ~MicCapture();

    MicCapture(const MicCapture&) = delete;
    MicCapture& operator=(const MicCapture&) = delete;

    // Returns false (and fills `error`) if no capture device is available.
    bool start(size_t sampleRate, streaming::RingBuffer& sink, std::string* error = nullptr);
    void stop();

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace audio
