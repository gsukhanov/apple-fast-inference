#pragma once

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
Tensor<_DType, _Device> avg_pool2d(const TensorView<_DType, _Device>& input,
                                   size_2_t kernel, size_2_t stride,
                                   size_2_t padding);
}  // namespace nn
}  // namespace fastinf

#include "../../../../src/nn/ops/avg_pool2d.hpp.inl"
