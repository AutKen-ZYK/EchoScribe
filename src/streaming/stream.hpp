#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

namespace streaming {

// Thread-safe FIFO of f32 samples (single producer / single consumer).
class RingBuffer {
public:
    explicit RingBuffer(size_t capacitySamples);

    void write(const float* data, size_t n);
    size_t read(float* dst, size_t n); // returns samples actually read
    size_t size() const;

private:
    mutable std::mutex mu_;
    std::vector<float> buf_;
    size_t head_ = 0; // next write position
    size_t count_ = 0;
};

struct TimedSegment {
    double t0 = 0.0;              // seconds from stream start
    std::vector<float> samples;   // 16 kHz mono
};

// Energy VAD with hysteresis: slices the stream into speech segments.
// 100 ms RMS blocks; speech starts above startRms, ends after
// silenceBlocksToEnd blocks below endRms; pre-roll keeps context before the
// first block. Over-long speech runs are force-flushed (maxSegmentSamples).
class Segmenter {
public:
    struct Config {
        float startRms = 0.012f;
        float endRms = 0.005f;
        size_t blockSamples = 1600;      // 100 ms at 16 kHz
        size_t silenceBlocksToEnd = 8;   // 800 ms
        size_t preRollBlocks = 4;        // 400 ms
        size_t minSegmentBlocks = 5;     // 500 ms
        size_t maxSegmentSamples = 28 * 16000;
    };

    Segmenter() : Segmenter(16000.0, Config()) {}
    explicit Segmenter(double sampleRate) : Segmenter(sampleRate, Config()) {}
    explicit Segmenter(double sampleRate, const Config& c);

    // Feed samples; completed speech segments are appended to out.
    void feed(const float* samples, size_t n, std::vector<TimedSegment>& out);
    // Flush any ongoing speech (call at shutdown).
    void flush(std::vector<TimedSegment>& out);

    bool inSpeech() const { return inSpeech_; }

private:
    void processBlock(std::vector<TimedSegment>& out);

    double sampleRate_;
    Config cfg_;
    std::vector<float> preRoll_;
    std::vector<float> pending_; // samples accumulated since stream start (up to block size)
    size_t pendingCount_ = 0;

    bool inSpeech_ = false;
    size_t silenceRun_ = 0;
    size_t segmentBlocks_ = 0;
    size_t totalSamples_ = 0; // samples fed so far (stream time base)
    size_t segmentStartSample_ = 0;
    std::vector<float> segment_;
};

} // namespace streaming
