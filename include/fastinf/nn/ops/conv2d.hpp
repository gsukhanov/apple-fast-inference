#pragma once

#include <functional>
#include <optional>

#include "fastinf/core/device.hpp"
#include "fastinf/core/dtype.hpp"
#include "fastinf/nn/ops/common.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class Tensor;

template <DType _DType, DeviceLikeType _Device>
class TensorView;

namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> conv2d(
    const TensorView<_DType, _Device>& input,
    const TensorView<_DType, _Device>& weight, size_2_t stride,
    size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<const TensorView<_DType, _Device>>>
        bias = std::nullopt);
}  // namespace nn
}  // namespace fastinf

#include "../../../../src/nn/ops/conv2d.hpp.inl"
