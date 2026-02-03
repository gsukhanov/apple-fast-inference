#include "fastinf/core/tensor_desc.hpp"

namespace fastinf {
TensorDesc::TensorDesc(Shape shape, Strides strides, std::int64_t offset)
    : shape_(std::move(shape)), strides_(std::move(strides)), offset_(offset) {
}

int TensorDesc::dim() const {
    return static_cast<int>(shape_.size());
}

std::int64_t TensorDesc::numel() const {
    if (shape_.empty()) {
        return 1;
    }

    std::int64_t n = 1;
    for (std::int64_t d : shape_) {
        n *= d;
    }

    return n;
}

inline Strides make_contiguous_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}

};  // namespace fastinf
