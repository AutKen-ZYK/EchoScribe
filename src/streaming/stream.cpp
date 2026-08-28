#include "streaming/stream.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace streaming {

RingBuffer::RingBuffer(size_t capacitySamples) : buf_(capacitySamples, 0.0f) {}

void RingBuffer::write(const float* data, size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t i = 0; i < n; ++i) {
        buf_[head_] = data[i];
        head_ = (head_ + 1) % buf_.size();
    }
    count_ = std::min(count_ + n, buf_.size());
}

size_t RingBuffer::read(float* dst, size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    const size_t got = std::min(n, count_);
    const size_t start = (head_ + buf_.size() - count_) % buf_.size();
    for (size_t i = 0; i < got; ++i) {
        dst[i] = buf_[(start + i) % buf_.size()];
    }
    count_ -= got;
    return got;
}

size_t RingBuffer::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return count_;
}

Segmenter::Segmenter(double sampleRate, const Config& c) : sampleRate_(sampleRate), cfg_(c) {
    if (cfg_.blockSamples == 0) throw std::runtime_error("segmenter: invalid config");
    preRoll_.reserve(cfg_.preRollBlocks * cfg_.blockSamples);
}

void Segmenter::processBlock(std::vector<TimedSegment>& out) {
    // RMS over this block
    double energy = 0.0;
    for (size_t i = 0; i < pendingCount_; ++i) {
        energy += static_cast<double>(pending_[i]) * pending_[i];
    }
    const double rms = std::sqrt(energy / static_cast<double>(pendingCount_));

    if (!inSpeech_) {
        // maintain pre-roll history
        const size_t preRollCap = cfg_.preRollBlocks * cfg_.blockSamples;
        for (size_t i = 0; i < pendingCount_; ++i) {
            preRoll_.push_back(pending_[i]);
        }
        if (preRoll_.size() > preRollCap) {
            preRoll_.erase(preRoll_.begin(),
                           preRoll_.begin() + static_cast<long>(preRoll_.size() - preRollCap));
        }
        if (rms >= cfg_.startRms) {
            inSpeech_ = true;
            silenceRun_ = 0;
            segmentBlocks_ = 1;
            segmentStartSample_ = totalSamples_ - pendingCount_ - preRoll_.size();
            segment_ = preRoll_;
            segment_.insert(segment_.end(), pending_.begin(), pending_.begin() + pendingCount_);
            preRoll_.clear();
        }
    } else {
        segment_.insert(segment_.end(), pending_.begin(), pending_.begin() + pendingCount_);
        ++segmentBlocks_;
        if (rms < cfg_.endRms) {
            ++silenceRun_;
        } else {
            silenceRun_ = 0;
        }

        const bool longEnough = segmentBlocks_ - silenceRun_ >= cfg_.minSegmentBlocks;
        const bool forceFlush = segment_.size() >= cfg_.maxSegmentSamples;
        if (forceFlush || (silenceRun_ >= cfg_.silenceBlocksToEnd && longEnough)) {
            if (longEnough) {
                TimedSegment seg;
                seg.t0 = static_cast<double>(segmentStartSample_) / sampleRate_;
                seg.samples = std::move(segment_);
                out.push_back(std::move(seg));
            }
            segment_.clear();
            segment_ = std::vector<float>();
            inSpeech_ = false;
            silenceRun_ = 0;
            segmentBlocks_ = 0;
        }
    }

    pendingCount_ = 0;
}

void Segmenter::feed(const float* samples, size_t n, std::vector<TimedSegment>& out) {
    size_t i = 0;
    while (i < n) {
        const size_t take = std::min(n - i, cfg_.blockSamples - pendingCount_);
        if (pending_.size() < cfg_.blockSamples) {
            pending_.resize(cfg_.blockSamples, 0.0f);
        }
        std::copy(samples + i, samples + i + take, pending_.begin() + pendingCount_);
        pendingCount_ += take;
        totalSamples_ += take;
        i += take;
        if (pendingCount_ == cfg_.blockSamples) {
            processBlock(out);
        }
    }
}

void Segmenter::flush(std::vector<TimedSegment>& out) {
    if (pendingCount_ > 0) {
        processBlock(out);
    }
    if (inSpeech_ && segmentBlocks_ >= cfg_.minSegmentBlocks && !segment_.empty()) {
        TimedSegment seg;
        seg.t0 = static_cast<double>(segmentStartSample_) / sampleRate_;
        seg.samples = std::move(segment_);
        out.push_back(std::move(seg));
    }
    segment_.clear();
    segmentBlocks_ = 0;
    inSpeech_ = false;
}

} // namespace streaming
