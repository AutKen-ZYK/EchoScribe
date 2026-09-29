#include "cli/transcribe.hpp"

#include "frontend/logmel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace transcribe {
namespace {

constexpr size_t kSampleRate = 16000;
constexpr size_t kWindowSamples = 480000; // 30 s
constexpr size_t kHop = 160;              // logmel hop at 16 kHz

std::vector<Segment> assembleSegments(const std::vector<size_t>& ids, const whisper::Tokenizer& tok,
                                      double timeOffset, const std::vector<float>& logprobs) {
    std::vector<Segment> out;
    std::string text;
    size_t startTs = 0;
    bool hasStart = false;
    size_t firstTokIdx = 0;

    auto flush = [&](size_t endTs, size_t tokIdxEnd) {
        if (!hasStart || text.empty()) return;
        double sum = 0;
        for (size_t i = firstTokIdx; i < tokIdxEnd; ++i) sum += logprobs[i];
        const double avg = sum / static_cast<double>(tokIdxEnd - firstTokIdx);
        if (avg >= -1.0) { // whisper logprob_threshold default
            Segment s;
            s.t0 = tok.timestampSeconds(startTs) + timeOffset;
            s.t1 = tok.timestampSeconds(endTs) + timeOffset;
            s.text = text;
            out.push_back(std::move(s));
        }
        text.clear();
    };

    for (size_t i = 0; i < ids.size(); ++i) {
        const size_t id = ids[i];
        if (tok.isTimestamp(id)) {
            flush(id, i);
            startTs = id;
            hasStart = true;
            firstTokIdx = i + 1;
        } else if (id == tok.eot()) {
            break;
        } else {
            text += tok.tokenText(id);
        }
    }
    // trailing segment without a closing timestamp: keep it, t1 = t0 + 2 s
    if (hasStart && !text.empty()) {
        Segment s;
        s.t0 = tok.timestampSeconds(startTs) + timeOffset;
        s.t1 = s.t0 + 2.0;
        s.text = text;
        out.push_back(std::move(s));
    }
    for (auto& s : out) {
        // whisper strips segment text for display
        const size_t b = s.text.find_first_not_of(" \t\n");
        const size_t e = s.text.find_last_not_of(" \t\n");
        s.text = (b == std::string::npos) ? "" : s.text.substr(b, e - b + 1);
    }
    return out;
}

} // namespace

Transcriber::Transcriber(const std::string& modelPath)
    : gguf_(gguf::File::open(modelPath)),
      tok_(gguf_),
      enc_(gguf_),
      dec_(gguf_) {
    // The mel bin count is a property of the model (80 for tiny..large-v2,
    // 128 for large-v3); take it from the encoder to keep the frontend and the
    // first conv in sync.
    nMels_ = enc_.nMels();
}

size_t Transcriber::detectLanguage(const tensor::Tensor& audio) {
    // One decoder pass with <|startoftranscript|> only; pick the most probable
    // language token (same principle as whisper's language detection).
    const std::vector<float> logits = dec_.forwardLogits(audio, {tok_.sot()});
    const size_t first = tok_.languageToken("en");
    const size_t last = tok_.transcribe(); // language block ends before <|translate|>
    size_t best = first;
    for (size_t id = first; id < last; ++id) {
        if (logits[id] > logits[best]) best = id;
    }
    return best;
}

std::vector<Segment> Transcriber::transcribeWindow(const float* pcm, size_t n, size_t langToken,
                                                   double timeOffset) {
    tensor::Tensor mel = frontend::logMelSpectrogram(pcm, n, kWindowSamples, nMels_);
    tensor::Tensor audio = enc_.forward(mel);

    std::vector<size_t> prompt = {tok_.sot(), langToken, tok_.transcribe()};
    std::vector<float> logprobs;
    std::vector<size_t> generated =
        dec_.generate(audio, tok_, prompt, /*noTimestamps=*/false, 448, &logprobs);
    return assembleSegments(generated, tok_, timeOffset, logprobs);
}

std::vector<Segment> Transcriber::transcribe(const float* pcm, size_t n, const std::string& lang) {
    if (n == 0) return {};
    std::vector<Segment> out;
    for (size_t offset = 0; offset < n; offset += kWindowSamples) {
        const size_t len = std::min(kWindowSamples, n - offset);

        // energy gate: skip windows that are essentially silent
        double energy = 0.0;
        for (size_t i = 0; i < len; ++i) energy += static_cast<double>(pcm[offset + i]) * pcm[offset + i];
        const double rms = std::sqrt(energy / static_cast<double>(len));
        if (rms < 1e-3) continue;

        tensor::Tensor mel = frontend::logMelSpectrogram(pcm + offset, len, kWindowSamples, nMels_);
        tensor::Tensor audio = enc_.forward(mel);

        size_t langToken;
        if (lang == "auto") {
            langToken = detectLanguage(audio);
        } else {
            langToken = tok_.languageToken(lang);
        }

        auto segs = transcribeWindow(pcm + offset, len, langToken,
                                     static_cast<double>(offset) / kSampleRate);
        out.insert(out.end(), segs.begin(), segs.end());
    }
    return out;
}

std::vector<Segment> Transcriber::transcribeFile(const std::string& wavPath,
                                                 const std::string& lang) {
    audio::WavData wav = audio::readWav(wavPath, kSampleRate);
    return transcribe(wav.samples.data(), wav.samples.size(), lang);
}

std::vector<Segment> Transcriber::transcribeSegment(const float* pcm, size_t n,
                                                    const std::string& lang) {
    if (n == 0) return {};

    // minimal padding: enough frames to cover the segment, even count
    size_t frames = (n + kHop - 1) / kHop;
    if (frames % 2 != 0) ++frames;
    const size_t target = frames * kHop;

    tensor::Tensor mel = frontend::logMelSpectrogram(pcm, n, target, nMels_);
    tensor::Tensor audio = enc_.forward(mel);

    size_t langToken;
    if (lang == "auto") {
        langToken = detectLanguage(audio);
    } else {
        langToken = tok_.languageToken(lang);
    }

    std::vector<size_t> prompt = {tok_.sot(), langToken, tok_.transcribe()};
    std::vector<float> logprobs;
    std::vector<size_t> generated =
        dec_.generate(audio, tok_, prompt, /*noTimestamps=*/false, 448, &logprobs);
    return assembleSegments(generated, tok_, 0.0, logprobs);
}

} // namespace transcribe
