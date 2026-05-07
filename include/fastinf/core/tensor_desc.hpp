#pragma once

#include <cstdint>
#include <vector>

namespace fastinf {
using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

struct TensorDesc {
    Shape shape_;
    Strides strides_;
    std::int64_t offset_ = 0;

    TensorDesc() = default;
    explicit TensorDesc(Shape shape, Strides strides, std::int64_t offset = 0);

    int dim() const;

    std::int64_t numel() const;
};

inline Strides make_contiguous_strides(const Shape& shapes);

};  // namespace fastinf

#include "../../../src/core/tensor_desc.hpp.inl"