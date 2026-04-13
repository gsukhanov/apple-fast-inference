#pragma once

#include <functional>
#include <optional>

#include "fastinf/core/device.hpp"
#include "fastinf/core/dtype.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class Tensor;

template <DType _DType, DeviceLikeType _Device>
class TensorView;

namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> conv2d(
    const TensorView<_DType, _Device>& input,
    const TensorView<_DType, _Device>& weight, std::int64_t stride_h,
    std::int64_t stride_w, std::int64_t padding_h, std::int64_t padding_w,
    std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t groups,
    std::optional<std::reference_wrapper<const TensorView<_DType, _Device>>> bias =
        std::nullopt);
}  // namespace nn
}  // namespace fastinf

#include "../../../../src/nn/ops/conv2d.hpp.inl"
