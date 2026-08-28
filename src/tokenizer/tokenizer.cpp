#include "tokenizer/tokenizer.hpp"

#include "gguf/loader.hpp"

#include <algorithm>
#include <stdexcept>

namespace whisper {
namespace {

// GPT-2 byte-to-unicode reverse table: token strings contain unicode
// codepoints; printable bytes map to themselves, the rest to 256+n.
std::unordered_map<int, unsigned char> buildCpToByte() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(cs.begin(), cs.end(), b) == cs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::unordered_map<int, unsigned char> m;
    m.reserve(256);
    for (size_t i = 0; i < bs.size(); ++i) m[cs[i]] = static_cast<unsigned char>(bs[i]);
    return m;
}

} // namespace

Tokenizer::Tokenizer(const gguf::File& f) : cpToByte_(buildCpToByte()) {
    auto& kv = f.kvMap();

    auto it = kv.find("tokenizer.ggml.tokens");
    if (it == kv.end()) throw std::runtime_error("tokenizer: no vocab in model");
    auto arr = std::get<std::shared_ptr<gguf::ArrayValue>>(it->second);
    tokens_.resize(arr->items.size());
    for (size_t i = 0; i < tokens_.size(); ++i) {
        tokens_[i] = std::get<std::string>(arr->items[i]);
        byName_[tokens_[i]] = i;
    }

    auto findByName = [&](const std::string& name, size_t fallback) {
        auto j = byName_.find(name);
        return j == byName_.end() ? fallback : j->second;
    };

    eot_ = findByName("<|endoftext|>", 50257);
    sot_ = findByName("<|startoftranscript|>", 50258);
    transcribe_ = findByName("<|transcribe|>", 50359);
    notimestamps_ = findByName("<|notimestamps|>", 50363);
    timestampBegin_ = findByName("<|0.00|>", nVocab() - 1501);

    if (auto u = kv.find("tokenizer.ggml.eos_token_id"); u != kv.end()) {
        eot_ = std::get<uint32_t>(u->second);
    }
    if (auto u = kv.find("stt.whisper.sot_token_id"); u != kv.end()) {
        sot_ = std::get<uint32_t>(u->second);
    }
    if (auto u = kv.find("stt.whisper.transcribe_token_id"); u != kv.end()) {
        transcribe_ = std::get<uint32_t>(u->second);
    }
    if (auto u = kv.find("stt.whisper.no_timestamps_token_id"); u != kv.end()) {
        notimestamps_ = std::get<uint32_t>(u->second);
    }

    if (auto s = kv.find("stt.whisper.suppress_tokens"); s != kv.end()) {
        auto a = std::get<std::shared_ptr<gguf::ArrayValue>>(s->second);
        for (auto& v : a->items) suppress_.push_back(std::get<int32_t>(v));
    }
    if (auto s = kv.find("stt.whisper.begin_suppress_tokens"); s != kv.end()) {
        auto a = std::get<std::shared_ptr<gguf::ArrayValue>>(s->second);
        for (auto& v : a->items) beginSuppress_.push_back(std::get<int32_t>(v));
    }
}

std::string Tokenizer::tokenText(size_t id) const {
    if (id >= tokens_.size()) throw std::runtime_error("tokenizer: token id out of range");
    const std::string& s = tokens_[id];
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int cp = -1;
        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
            i += 3;
        } else {
            cp = c;
            i += 1;
        }
        auto it = cpToByte_.find(cp);
        if (it != cpToByte_.end()) {
            out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<size_t>& ids) const {
    std::string out;
    for (size_t id : ids) {
        if (isSpecial(id)) continue;
        out += tokenText(id);
    }
    return out;
}

size_t Tokenizer::languageToken(const std::string& code) const {
    auto it = byName_.find("<|" + code + "|>");
    if (it == byName_.end()) throw std::runtime_error("tokenizer: unknown language " + code);
    return it->second;
}

} // namespace whisper
