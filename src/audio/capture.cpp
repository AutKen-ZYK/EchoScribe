#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio/capture.hpp"

#include <stdexcept>

namespace audio {

struct MicCapture::Impl {
    ma_device device{};
    streaming::RingBuffer* sink = nullptr;
    bool started = false;

    static void callback(ma_device* pDevice, void* pOutput, const void* pInput,
                         ma_uint32 frameCount) {
        (void)pOutput;
        auto* self = static_cast<Impl*>(pDevice->pUserData);
        if (self && self->sink && pInput) {
            self->sink->write(static_cast<const float*>(pInput), frameCount);
        }
    }
};

MicCapture::~MicCapture() {
    stop();
}

bool MicCapture::start(size_t sampleRate, streaming::RingBuffer& sink, std::string* error) {
    impl_ = new Impl();
    impl_->sink = &sink;

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = static_cast<ma_uint32>(sampleRate); // miniaudio resamples if needed
    config.periodSizeInFrames = 1600;                       // ~100 ms
    config.pUserData = impl_;
    config.dataCallback = &Impl::callback;

    if (ma_device_init(nullptr, &config, &impl_->device) != MA_SUCCESS) {
        if (error) *error = "no capture device found";
        delete impl_;
        impl_ = nullptr;
        return false;
    }
    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        if (error) *error = "failed to start capture device";
        ma_device_uninit(&impl_->device);
        delete impl_;
        impl_ = nullptr;
        return false;
    }
    impl_->started = true;
    return true;
}

void MicCapture::stop() {
    if (impl_ && impl_->started) {
        ma_device_uninit(&impl_->device);
    }
    delete impl_;
    impl_ = nullptr;
}

} // namespace audio
