#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensor {

// Contiguous row-major f32 tensor. dims[0] is the fastest-varying dimension
// (ggml convention).
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape);

    static Tensor fromF32(const void* src, std::vector<size_t> shape);
    static Tensor fromF16(const void* src, std::vector<size_t> shape);

    const std::vector<size_t>& shape() const { return shape_; }
    std::vector<size_t> strides() const;
    size_t size() const { return data_.size(); }
    size_t ndim() const { return shape_.size(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    Tensor reshape(std::vector<size_t> newShape) const;

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
};

float f16ToF32(uint16_t h);

} // namespace tensor
