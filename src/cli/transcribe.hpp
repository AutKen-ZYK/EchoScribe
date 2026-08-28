#pragma once

#include "audio/wav.hpp"
#include "decoder/decoder.hpp"
#include "encoder/encoder.hpp"
#include "gguf/gguf.hpp"
#include "tokenizer/tokenizer.hpp"

#include <string>
#include <vector>

namespace transcribe {

struct Segment {
    double t0 = 0.0; // seconds
    double t1 = 0.0; // seconds
    std::string text;
};

// End-to-end pipeline: 16 kHz mono PCM -> segments with timestamps.
// Processes audio in 30 s windows (padded with zeros at the tail).
class Transcriber {
public:
    explicit Transcriber(const std::string& modelPath);

    std::vector<Segment> transcribe(const float* pcm, size_t n, const std::string& lang = "auto");

    std::vector<Segment> transcribeFile(const std::string& wavPath,
                                        const std::string& lang = "auto");

    // Real-time path: transcribe one short segment with minimal padding
    // (no 30 s window), timestamps relative to segment start.
    std::vector<Segment> transcribeSegment(const float* pcm, size_t n,
                                           const std::string& lang = "auto");

    const whisper::Tokenizer& tokenizer() const { return tok_; }

private:
    std::vector<Segment> transcribeWindow(const float* pcm, size_t n, size_t langToken,
                                          double timeOffset);
    size_t detectLanguage(const tensor::Tensor& audio);

    gguf::File gguf_;
    whisper::Tokenizer tok_;
    whisper::Encoder enc_;
    whisper::Decoder dec_;
};

} // namespace transcribe
