#pragma once

#include "gguf/gguf.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace whisper {

// GPT-2 byte-level BPE tokenizer over the vocab stored in the GGUF
// (tokenizer.ggml.tokens). Decoding only: text -> ids is not needed for
// transcription. Token strings use the GPT-2 byte-to-unicode mapping
// ("ĠAnd" for " And") and are converted back to raw UTF-8 bytes here.
class Tokenizer {
public:
    explicit Tokenizer(const gguf::File& f);

    size_t nVocab() const { return tokens_.size(); }

    // Decoded raw bytes of one token (special tokens decode to their literal
    // "<|...|>" text).
    std::string tokenText(size_t id) const;

    // Concatenated text of non-special tokens (timestamps and other specials
    // are skipped).
    std::string decode(const std::vector<size_t>& ids) const;

    // Special tokens, resolved from the vocab/config in the file.
    size_t sot() const { return sot_; }
    size_t eot() const { return eot_; }
    size_t transcribe() const { return transcribe_; }
    size_t notimestamps() const { return notimestamps_; }
    size_t timestampBegin() const { return timestampBegin_; }
    size_t languageToken(const std::string& code) const; // e.g. "en" -> 50259

    bool isSpecial(size_t id) const { return id >= eot_; }
    bool isTimestamp(size_t id) const { return id >= timestampBegin_; }
    double timestampSeconds(size_t id) const {
        return static_cast<double>(id - timestampBegin_) * 0.02;
    }

    const std::vector<int32_t>& suppressTokens() const { return suppress_; }
    const std::vector<int32_t>& beginSuppressTokens() const { return beginSuppress_; }

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, size_t> byName_;
    std::unordered_map<int, unsigned char> cpToByte_;
    std::vector<int32_t> suppress_;
    std::vector<int32_t> beginSuppress_;
    size_t sot_ = 0, eot_ = 0, transcribe_ = 0, notimestamps_ = 0, timestampBegin_ = 0;
};

} // namespace whisper
