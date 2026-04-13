#include "fastinf/nn/avg_pool2d.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
AvgPool2d<_DType, _Device>::AvgPool2d(size_2_t kernel, size_2_t stride,
                                      size_2_t padding)
    : kernel_(kernel), stride_(stride), padding_(padding) {
    if (kernel_.h <= 0 || kernel_.w <= 0) {
        throw std::invalid_argument("Kernel dimensions must be positive");
    }
    if (stride_.h <= 0 || stride_.w <= 0) {
        throw std::invalid_argument("Stride dimensions must be positive");
    }
    if (padding_.h < 0 || padding_.w < 0) {
        throw std::invalid_argument("Padding dimensions must be non-negative");
    }
}

template <DType _DType, DeviceLikeType _Device>
AvgPool2d<_DType, _Device>::AvgPool2d(std::int64_t kernel, std::int64_t stride,
                                      std::int64_t padding)
    : AvgPool2d({kernel, kernel}, {stride, stride}, {padding, padding}) {
}

template <DType _DType, DeviceLikeType _Device>
std::string AvgPool2d<_DType, _Device>::name() const {
    return "nn.AvgPool2d";
}

template <DType _DType, DeviceLikeType _Device>
typename AvgPool2d<_DType, _Device>::tensor_t
AvgPool2d<_DType, _Device>::forward(const tensor_t& input) const {
    return nn::avg_pool2d<_DType, _Device>(
        input.view(), kernel_.h, kernel_.w, stride_.h, stride_.w, padding_.h,
        padding_.w);
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t AvgPool2d<_DType, _Device>::output_height(
    std::int64_t input_h) const {
    const auto num = input_h + 2 * padding_.h - kernel_.h;
    if (num < 0) {
        throw std::invalid_argument(
            "Kernel height larger than padded input height");
    }

    return num / stride_.h + 1;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t AvgPool2d<_DType, _Device>::output_width(
    std::int64_t input_w) const {
    const auto num = input_w + 2 * padding_.w - kernel_.w;
    if (num < 0) {
        throw std::invalid_argument(
            "Kernel width larger than padded input width");
    }

    return num / stride_.w + 1;
}
}  // namespace nn
}  // namespace fastinf
