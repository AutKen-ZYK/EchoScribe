#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace gguf {

// ggml tensor element types (subset relevant to whisper models).
enum class ElementType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    Unknown = 0xFFFFFFFFu,
};

const char* elementTypeName(ElementType t);
size_t elementSize(ElementType t); // 0 for types we do not handle yet

enum class ValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

using ScalarValue = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
                                 bool, std::string, uint64_t, int64_t, double>;

struct ArrayValue {
    ValueType elemType = ValueType::UINT8;
    std::vector<ScalarValue> items;
};

using Value = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, bool,
                           std::string, uint64_t, int64_t, double, std::shared_ptr<ArrayValue>>;

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims; // dims[0] = fastest varying
    ElementType type = ElementType::Unknown;
    uint64_t offset = 0; // relative to the start of the data section
    uint64_t nBytes = 0; // 0 for unsupported (quantized) types
};

// Parsed GGUF file backed by an mmap of the original file. Tensor data
// pointers remain valid for the lifetime of the File object.
class File {
public:
    File();
    ~File();
    File(File&&) noexcept;
    File& operator=(File&&) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    static File open(const std::string& path); // throws std::runtime_error

    uint32_t version() const { return version_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t tensorCount() const { return tensors_.size(); }
    size_t kvCount() const;

    bool hasKV(const std::string& key) const;
    const Value& kv(const std::string& key) const; // throws if missing
    const std::unordered_map<std::string, Value>& kvMap() const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* findTensor(const std::string& name) const;

    const void* tensorData(const TensorInfo& t) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    std::vector<TensorInfo> tensors_;
};

} // namespace gguf
