#include "gguf/loader.hpp"

#include <stdexcept>

namespace gguf {

tensor::Tensor loadTensorF32(const File& f, const std::string& name) {
    const TensorInfo* info = f.findTensor(name);
    if (!info) throw std::runtime_error("gguf: tensor not found: " + name);

    // ggml dims {d0, d1, ...} (d0 fastest) -> torch shape {..., d1, d0}.
    std::vector<size_t> shape(info->dims.rbegin(), info->dims.rend());
    const void* src = f.tensorData(*info);

    if (info->type == ElementType::F32) {
        return tensor::Tensor::fromF32(src, std::move(shape));
    }
    if (info->type == ElementType::F16) {
        return tensor::Tensor::fromF16(src, std::move(shape));
    }
    throw std::runtime_error("gguf: tensor '" + name + "' has unsupported type " +
                             elementTypeName(info->type));
}

} // namespace gguf
