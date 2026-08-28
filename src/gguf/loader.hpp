#pragma once

#include "gguf/gguf.hpp"
#include "tensor/tensor.hpp"

#include <string>

namespace gguf {

// Load a tensor by name, converting f16 weights to f32. Throws std::runtime_error
// if the tensor is missing or has an unsupported (quantized) type.
tensor::Tensor loadTensorF32(const File& f, const std::string& name);

} // namespace gguf
