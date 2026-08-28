#include "audio/wav.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace audio {
namespace {

uint16_t readU16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t readU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

std::vector<float> resampleLinear(const std::vector<float>& in, size_t srcRate, size_t dstRate) {
    if (srcRate == dstRate || in.empty()) return in;
    const double ratio = static_cast<double>(srcRate) / static_cast<double>(dstRate);
    const size_t outN = static_cast<size_t>(static_cast<double>(in.size()) / ratio);
    std::vector<float> out(outN);
    for (size_t i = 0; i < outN; ++i) {
        const double src = static_cast<double>(i) * ratio;
        const size_t i0 = static_cast<size_t>(src);
        const size_t i1 = std::min(i0 + 1, in.size() - 1);
        const double frac = src - static_cast<double>(i0);
        out[i] = static_cast<float>(in[i0] * (1.0 - frac) + in[i1] * frac);
    }
    return out;
}

} // namespace

WavData readWav(const std::string& path, size_t targetRate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("wav: cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 44 || std::memcmp(buf.data(), "RIFF", 4) != 0 ||
        std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("wav: not a RIFF/WAVE file: " + path);
    }

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data = nullptr;
    size_t dataBytes = 0;

    size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        const char* id = reinterpret_cast<const char*>(buf.data() + pos);
        const uint32_t size = readU32(buf.data() + pos + 4);
        const uint8_t* body = buf.data() + pos + 8;
        if (size > buf.size() - pos - 8) {
            throw std::runtime_error("wav: truncated chunk in " + path);
        }
        if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16) {
            format = readU16(body);
            channels = readU16(body + 2);
            rate = readU32(body + 4);
            bits = readU16(body + 14);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = body;
            dataBytes = size;
        }
        pos += 8 + size + (size & 1); // chunks are word-aligned
    }

    if (data == nullptr || channels == 0 || rate == 0) {
        throw std::runtime_error("wav: missing fmt/data chunk in " + path);
    }

    const size_t frames = dataBytes / (channels * (bits / 8));
    std::vector<float> mono(frames);

    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (size_t c = 0; c < channels; ++c) {
            const uint8_t* p = data + (i * channels + c) * (bits / 8);
            double v = 0.0;
            if (format == 1 && bits == 16) {
                v = static_cast<double>(static_cast<int16_t>(readU16(p))) / 32768.0;
            } else if (format == 1 && bits == 32) {
                int32_t s;
                std::memcpy(&s, p, 4);
                v = static_cast<double>(s) / 2147483648.0;
            } else if (format == 3 && bits == 32) {
                float fv;
                std::memcpy(&fv, p, 4);
                v = fv;
            } else {
                throw std::runtime_error("wav: unsupported format (format=" +
                                         std::to_string(format) + " bits=" + std::to_string(bits) +
                                         ")");
            }
            acc += v;
        }
        mono[i] = static_cast<float>(acc / channels);
    }

    return WavData{resampleLinear(mono, rate, targetRate), targetRate};
}

} // namespace audio
