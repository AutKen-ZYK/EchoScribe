#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gguf/gguf.hpp"
#include "tokenizer/tokenizer.hpp"

#include <string>
#include <vector>

#ifndef ECHOSCRIBE_MODELS_DIR
#define ECHOSCRIBE_MODELS_DIR "models"
#endif

using whisper::Tokenizer;

TEST_CASE("tokenizer special tokens and byte decoding", "[tokenizer]") {
    gguf::File f = gguf::File::open(ECHOSCRIBE_MODELS_DIR "/whisper-tiny-F16.gguf");
    Tokenizer tok(f);

    REQUIRE(tok.nVocab() == 51865);
    REQUIRE(tok.sot() == 50258);
    REQUIRE(tok.eot() == 50257);
    REQUIRE(tok.transcribe() == 50359);
    REQUIRE(tok.notimestamps() == 50363);
    REQUIRE(tok.timestampBegin() == 50364);
    REQUIRE(tok.languageToken("en") == 50259);
    REQUIRE(tok.languageToken("zh") == 50260); // second language in whisper's order

    REQUIRE(tok.isSpecial(50257));
    REQUIRE(tok.isTimestamp(50364));
    REQUIRE(!tok.isTimestamp(50256));
    REQUIRE(tok.timestampSeconds(50364) == Catch::Approx(0.0));
    REQUIRE(tok.timestampSeconds(50394) == Catch::Approx(0.6).margin(1e-9)); // +30 * 0.02

    // GPT-2 byte-level decoding: 'And' is a single token at id 400 whose raw
    // string is "ĠAnd" (mapped leading space)
    REQUIRE(tok.tokenText(400) == " And");
    // 'Ġ' (mapped space): "Ġso" etc.
    REQUIRE(tok.tokenText(370) == " so");
    // ASCII bytes are their own tokens: '!' = 0
    REQUIRE(tok.tokenText(0) == "!");
    // special tokens keep their literal text
    REQUIRE(tok.tokenText(50257) == "<|endoftext|>");

    SECTION("decode skips specials and joins bytes") {
        std::vector<size_t> ids = {400, 370, 452, 7177, 6280, 50257, 13};
        REQUIRE(tok.decode(ids) == " And so my fellow Americans.");
    }

    SECTION("suppress lists loaded from config") {
        REQUIRE(tok.beginSuppressTokens().size() == 2);
        REQUIRE(tok.beginSuppressTokens()[0] == 220);  // space
        REQUIRE(tok.beginSuppressTokens()[1] == 50257); // eot
        REQUIRE(tok.suppressTokens().size() == 88);
    }
}
