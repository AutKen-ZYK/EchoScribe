#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "gguf/gguf.hpp"
#include "gguf/loader.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

// Scratch path inside the platform temp directory: hardcoding a fixed path
// (e.g. /tmp/opencode) breaks the suite on any machine that lacks that dir.
std::string tempPath(const char* name) {
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec || dir.empty()) dir = std::filesystem::current_path(ec);
    return (dir / name).string();
}

// Minimal GGUF v3 writer used to build synthetic test files in memory.
struct Buf {
    std::vector<uint8_t> b;
    template <typename T>
    void put(const T& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void putStr(const std::string& s) {
        const uint64_t n = s.size();
        put(n);
        b.insert(b.end(), s.begin(), s.end());
    }
    void write(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    }
};

// Builds a file with: kv (u32 alignment, string name, array of 2 strings),
// tensors: "a.weight" f32 {2,3}, "b.bias" f16 {4}.
std::string writeTestGguf(const std::string& path) {
    Buf b;
    const uint32_t magic = 0x46554747u; // "GGUF"
    b.put(magic);
    b.put(static_cast<uint32_t>(3)); // version
    b.put(static_cast<uint64_t>(2)); // tensor count
    b.put(static_cast<uint64_t>(3)); // kv count

    b.putStr("general.alignment");
    b.put(static_cast<uint32_t>(4)); // ValueType::UINT32
    b.put(static_cast<uint32_t>(16)); // alignment value

    b.putStr("general.name");
    b.put(static_cast<uint32_t>(8)); // ValueType::STRING
    b.putStr("test-model");

    b.putStr("test.tokens");
    b.put(static_cast<uint32_t>(9)); // ValueType::ARRAY
    b.put(static_cast<uint32_t>(8)); // of STRING
    b.put(static_cast<uint64_t>(2));
    b.putStr("hello");
    b.putStr("world");

    // tensor infos (offsets are relative to data start, which follows alignment)
    b.putStr("a.weight");
    b.put(static_cast<uint32_t>(2));        // n_dims
    b.put(static_cast<uint64_t>(2));        // dims[0]
    b.put(static_cast<uint64_t>(3));        // dims[1]
    b.put(static_cast<uint32_t>(0));        // F32
    b.put(static_cast<uint64_t>(0));        // offset

    b.putStr("b.bias");
    b.put(static_cast<uint32_t>(1));
    b.put(static_cast<uint64_t>(4));
    b.put(static_cast<uint32_t>(1)); // F16
    b.put(static_cast<uint64_t>(6 * 4));

    // pad header to alignment (16); data starts right after tensor infos
    const uint64_t headerEnd = static_cast<uint64_t>(b.b.size());
    const uint64_t aligned = (headerEnd + 15) / 16 * 16;
    b.b.insert(b.b.end(), aligned - headerEnd, 0);

    const float aData[6] = {1.f, -2.f, 3.f, 4.f, -5.5f, 6.f};
    b.b.insert(b.b.end(), reinterpret_cast<const uint8_t*>(aData),
               reinterpret_cast<const uint8_t*>(aData) + sizeof(aData));
    const uint16_t bData[4] = {0x3C00, 0xC000, 0x0000, 0x3555}; // 1, -2, 0, 0.3333
    b.b.insert(b.b.end(), reinterpret_cast<const uint8_t*>(bData),
               reinterpret_cast<const uint8_t*>(bData) + sizeof(bData));

    b.write(path);
    return path;
}

} // namespace

TEST_CASE("gguf parser: header, kv, tensors", "[gguf]") {
    const std::string path = tempPath("echoscribe_test.gguf");
    writeTestGguf(path);

    gguf::File f = gguf::File::open(path);

    REQUIRE(f.version() == 3);
    REQUIRE(f.tensorCount() == 2);
    REQUIRE(f.kvCount() == 3);
    REQUIRE(f.alignment() == 16);

    REQUIRE(f.hasKV("general.name"));
    REQUIRE(std::get<std::string>(f.kv("general.name")) == "test-model");
    REQUIRE(std::get<uint32_t>(f.kv("general.alignment")) == 16);

    const auto& tokens = std::get<std::shared_ptr<gguf::ArrayValue>>(f.kv("test.tokens"));
    REQUIRE(tokens->items.size() == 2);
    REQUIRE(std::get<std::string>(tokens->items[0]) == "hello");
    REQUIRE(std::get<std::string>(tokens->items[1]) == "world");

    const gguf::TensorInfo* a = f.findTensor("a.weight");
    REQUIRE(a != nullptr);
    REQUIRE(a->dims == std::vector<uint64_t>{2, 3});
    REQUIRE(a->type == gguf::ElementType::F32);
    REQUIRE(a->nBytes == 24);

    const gguf::TensorInfo* missing = f.findTensor("nope");
    REQUIRE(missing == nullptr);
}

TEST_CASE("gguf loader: f32 and f16 to f32", "[gguf]") {
    const std::string path = tempPath("echoscribe_test.gguf");
    writeTestGguf(path);
    gguf::File f = gguf::File::open(path);

    tensor::Tensor a = gguf::loadTensorF32(f, "a.weight");
    REQUIRE(a.size() == 6);
    // ggml dims {2,3} -> torch shape {3,2}
    REQUIRE(a.shape() == std::vector<size_t>{3, 2});
    REQUIRE_THAT(a.data()[0], WithinAbs(1.0f, 1e-6));
    REQUIRE_THAT(a.data()[1], WithinAbs(-2.0f, 1e-6));
    REQUIRE_THAT(a.data()[4], WithinAbs(-5.5f, 1e-6));

    tensor::Tensor b = gguf::loadTensorF32(f, "b.bias");
    REQUIRE(b.size() == 4);
    REQUIRE_THAT(b.data()[0], WithinAbs(1.0f, 1e-6));
    REQUIRE_THAT(b.data()[1], WithinAbs(-2.0f, 1e-6));
    REQUIRE_THAT(b.data()[2], WithinAbs(0.0f, 1e-6));
    REQUIRE_THAT(b.data()[3], WithinAbs(0.33325195f, 1e-5));

    REQUIRE_THROWS(gguf::loadTensorF32(f, "does.not.exist"));
}

TEST_CASE("gguf parser: rejects non-gguf file", "[gguf]") {
    const std::string path = tempPath("echoscribe_bad.gguf");
    {
        std::ofstream f(path, std::ios::binary);
        REQUIRE(f.good()); // fail loudly rather than passing for the wrong reason
        f << "this is not a gguf file, just some text padding padding";
        f.close();
    }
    REQUIRE(std::filesystem::exists(path));
    REQUIRE(std::filesystem::file_size(path) >= 8); // must pass the size check
    REQUIRE_THROWS(gguf::File::open(path));
    std::filesystem::remove(path);
}

TEST_CASE("gguf parser: rejects a missing file", "[gguf]") {
    REQUIRE_THROWS(gguf::File::open(tempPath("echoscribe_does_not_exist.gguf")));
}

// Real-model smoke test: skipped automatically when no model is present.
#ifndef ECHOSCRIBE_MODELS_DIR
#define ECHOSCRIBE_MODELS_DIR "models"
#endif

TEST_CASE("gguf parser: real whisper tiny model", "[gguf][model]") {
    const char* candidates[] = {ECHOSCRIBE_MODELS_DIR "/whisper-tiny-F16.gguf"};
    std::string found;
    for (const char* c : candidates) {
        std::ifstream f(c, std::ios::binary | std::ios::ate);
        if (f.good() && f.tellg() > 0) {
            found = c;
            break;
        }
    }
    if (found.empty()) {
        WARN("no model file found, skipping real-model smoke test");
        return;
    }

    gguf::File f = gguf::File::open(found);
    REQUIRE(f.tensorCount() > 100);

    // llama.cpp conversion naming: enc.conv.0.* (conv1), enc.blocks.N.*, etc.
    // ggml stores dims reversed vs torch: conv weight torch [out=384, in=80, k=3]
    // is stored as dims {3, 80, 384} with dims[0] fastest; the linear memory
    // layout matches torch contiguous.
    const gguf::TensorInfo* conv1 = f.findTensor("enc.conv.0.weight");
    REQUIRE(conv1 != nullptr);
    REQUIRE(conv1->dims[0] == 3);
    REQUIRE(conv1->dims[1] == 80);  // mel bins
    REQUIRE(conv1->dims[2] == 384); // tiny d_model

    tensor::Tensor w = gguf::loadTensorF32(f, "enc.conv.0.weight");
    REQUIRE(w.size() == 80 * 384 * 3);
    double sum = 0;
    for (size_t i = 0; i < w.size(); ++i) sum += w.data()[i];
    const double mean = sum / static_cast<double>(w.size());
    REQUIRE(std::abs(mean) < 0.1); // weight init is small
}
