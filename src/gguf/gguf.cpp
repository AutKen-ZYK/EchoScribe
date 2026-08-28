#include "gguf/gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace gguf {
namespace {

constexpr uint32_t kMagic = 0x46554747u; // "GGUF" little-endian

uint64_t alignUp(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

struct Reader {
    const uint8_t* p;
    const uint8_t* end;

    void need(size_t n) const {
        if (static_cast<size_t>(end - p) < n) {
            throw std::runtime_error("gguf: unexpected end of file while parsing");
        }
    }

    template <typename T>
    T scalar() {
        static_assert(std::is_trivially_copyable_v<T>);
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }

    std::string str() {
        const uint64_t len = scalar<uint64_t>();
        need(len);
        std::string s(reinterpret_cast<const char*>(p), len);
        p += len;
        return s;
    }
};

ScalarValue parseScalar(Reader& r, ValueType type) {
    switch (type) {
        case ValueType::UINT8: return r.scalar<uint8_t>();
        case ValueType::INT8: return r.scalar<int8_t>();
        case ValueType::UINT16: return r.scalar<uint16_t>();
        case ValueType::INT16: return r.scalar<int16_t>();
        case ValueType::UINT32: return r.scalar<uint32_t>();
        case ValueType::INT32: return r.scalar<int32_t>();
        case ValueType::FLOAT32: return r.scalar<float>();
        case ValueType::BOOL: return r.scalar<uint8_t>() != 0;
        case ValueType::STRING: return r.str();
        case ValueType::UINT64: return r.scalar<uint64_t>();
        case ValueType::INT64: return r.scalar<int64_t>();
        case ValueType::FLOAT64: return r.scalar<double>();
    }
    throw std::runtime_error("gguf: unknown metadata value type");
}

Value parseValue(Reader& r, ValueType type) {
    if (type == ValueType::ARRAY) {
        auto arr = std::make_shared<ArrayValue>();
        arr->elemType = static_cast<ValueType>(r.scalar<uint32_t>());
        const uint64_t count = r.scalar<uint64_t>();
        arr->items.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            arr->items.push_back(parseScalar(r, arr->elemType));
        }
        return arr;
    }
    Value v;
    std::visit([&v](auto&& x) { v = std::move(x); }, parseScalar(r, type));
    return v;
}

} // namespace

const char* elementTypeName(ElementType t) {
    switch (t) {
        case ElementType::F32: return "f32";
        case ElementType::F16: return "f16";
        case ElementType::Q4_0: return "q4_0";
        case ElementType::Q4_1: return "q4_1";
        case ElementType::Q5_0: return "q5_0";
        case ElementType::Q5_1: return "q5_1";
        case ElementType::Q8_0: return "q8_0";
        case ElementType::Q2_K: return "q2_k";
        case ElementType::Q3_K: return "q3_k";
        case ElementType::Q4_K: return "q4_k";
        case ElementType::Q5_K: return "q5_k";
        case ElementType::Q6_K: return "q6_k";
        case ElementType::Q8_K: return "q8_k";
        case ElementType::Unknown: return "unknown";
    }
    return "unknown";
}

size_t elementSize(ElementType t) {
    switch (t) {
        case ElementType::F32: return 4;
        case ElementType::F16: return 2;
        default: return 0; // block-quantized types: handled via block tables
    }
}

class File::Impl {
public:
    int fd = -1;
    void* map = nullptr;
    size_t mapSize = 0;
    uint64_t dataStart = 0;
    std::unordered_map<std::string, Value> kv;
};

File::File() = default;
File::~File() {
    if (impl_ && impl_->map) {
        munmap(impl_->map, impl_->mapSize);
    }
    if (impl_ && impl_->fd >= 0) {
        close(impl_->fd);
    }
}
File::File(File&&) noexcept = default;
File& File::operator=(File&&) noexcept = default;

File File::open(const std::string& path) {
    File f;
    f.impl_ = std::make_unique<Impl>();
    Impl& impl = *f.impl_;

    impl.fd = ::open(path.c_str(), O_RDONLY);
    if (impl.fd < 0) throw std::runtime_error("gguf: cannot open " + path);

    struct stat st{};
    if (::fstat(impl.fd, &st) != 0 || st.st_size < 8) {
        throw std::runtime_error("gguf: cannot stat or empty file " + path);
    }
    impl.mapSize = static_cast<size_t>(st.st_size);
    impl.map = ::mmap(nullptr, impl.mapSize, PROT_READ, MAP_PRIVATE, impl.fd, 0);
    if (impl.map == MAP_FAILED) {
        impl.map = nullptr;
        throw std::runtime_error("gguf: mmap failed for " + path);
    }

    Reader r{static_cast<const uint8_t*>(impl.map), static_cast<const uint8_t*>(impl.map) + impl.mapSize};

    if (r.scalar<uint32_t>() != kMagic) {
        throw std::runtime_error("gguf: bad magic, not a GGUF file: " + path);
    }
    f.version_ = r.scalar<uint32_t>();
    if (f.version_ < 2 || f.version_ > 3) {
        throw std::runtime_error("gguf: unsupported version " + std::to_string(f.version_));
    }

    const uint64_t tensorCount = r.scalar<uint64_t>();
    const uint64_t kvCount = r.scalar<uint64_t>();

    f.tensors_.reserve(static_cast<size_t>(tensorCount));
    impl.kv.reserve(static_cast<size_t>(kvCount));

    for (uint64_t i = 0; i < kvCount; ++i) {
        std::string key = r.str();
        const auto vt = static_cast<ValueType>(r.scalar<uint32_t>());
        impl.kv.emplace(std::move(key), parseValue(r, vt));
    }

    for (uint64_t i = 0; i < tensorCount; ++i) {
        TensorInfo t;
        t.name = r.str();
        const uint32_t ndims = r.scalar<uint32_t>();
        if (ndims == 0 || ndims > 4) {
            throw std::runtime_error("gguf: tensor '" + t.name + "' has unsupported rank");
        }
        t.dims.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d) t.dims[d] = r.scalar<uint64_t>();
        t.type = static_cast<ElementType>(r.scalar<uint32_t>());
        t.offset = r.scalar<uint64_t>();

        const size_t es = elementSize(t.type);
        if (es != 0) {
            uint64_t numel = 1;
            for (uint64_t d : t.dims) numel *= d;
            t.nBytes = numel * es;
        }
        f.tensors_.push_back(std::move(t));
    }

    // KV may override the data-section alignment.
    if (impl.kv.count("general.alignment")) {
        const auto& v = impl.kv.at("general.alignment");
        if (auto* a = std::get_if<uint32_t>(&v)) f.alignment_ = *a;
    }

    const uint64_t parsed = static_cast<uint64_t>(r.p - static_cast<const uint8_t*>(impl.map));
    impl.dataStart = alignUp(parsed, f.alignment_);

    for (const TensorInfo& t : f.tensors_) {
        if (t.nBytes == 0) continue; // unsupported quant type: skip bounds check
        if (impl.dataStart + t.offset + t.nBytes > impl.mapSize) {
            throw std::runtime_error("gguf: tensor '" + t.name + "' out of file bounds");
        }
    }

    return f;
}

bool File::hasKV(const std::string& key) const {
    return impl_->kv.count(key) != 0;
}

size_t File::kvCount() const {
    return impl_->kv.size();
}

const std::unordered_map<std::string, Value>& File::kvMap() const {
    return impl_->kv;
}

const Value& File::kv(const std::string& key) const {
    auto it = impl_->kv.find(key);
    if (it == impl_->kv.end()) throw std::runtime_error("gguf: missing metadata key " + key);
    return it->second;
}

const TensorInfo* File::findTensor(const std::string& name) const {
    for (const TensorInfo& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

const void* File::tensorData(const TensorInfo& t) const {
    return static_cast<const uint8_t*>(impl_->map) + impl_->dataStart + t.offset;
}

} // namespace gguf
