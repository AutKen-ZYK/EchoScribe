#include "tensor/tensor.hpp"

#include <cstring>
#include <stdexcept>

namespace tensor {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    size_t n = 1;
    for (size_t d : shape_) n *= d;
    data_.assign(n, 0.0f);
}

static size_t product(const std::vector<size_t>& dims) {
    size_t n = 1;
    for (size_t d : dims) n *= d;
    return n;
}

Tensor Tensor::fromF32(const void* src, std::vector<size_t> shape) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.data_.resize(product(t.shape_));
    std::memcpy(t.data_.data(), src, t.data_.size() * sizeof(float));
    return t;
}

Tensor Tensor::fromF16(const void* src, std::vector<size_t> shape) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.data_.resize(product(t.shape_));
    const uint16_t* h = static_cast<const uint16_t*>(src);
    for (size_t i = 0; i < t.data_.size(); ++i) t.data_[i] = f16ToF32(h[i]);
    return t;
}

std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> s(shape_.size(), 1);
    for (size_t i = shape_.size(); i-- > 1;) s[i - 1] = s[i] * shape_[i];
    return s;
}

Tensor Tensor::reshape(std::vector<size_t> newShape) const {
    if (product(newShape) != data_.size()) {
        throw std::runtime_error("reshape: element count mismatch");
    }
    Tensor t;
    t.shape_ = std::move(newShape);
    t.data_ = data_;
    return t;
}

float f16ToF32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

} // namespace tensor
