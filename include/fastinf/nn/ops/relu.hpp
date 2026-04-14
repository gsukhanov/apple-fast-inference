#pragma once

#include "fastinf/core/device.hpp"
#include "fastinf/core/dtype.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class Tensor;

template <DType _DType, DeviceLikeType _Device>
class TensorView;

namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> relu(const TensorView<_DType, _Device>& input);
}  // namespace nn
}  // namespace fastinf

#include "../../../../src/nn/ops/relu.hpp.inl"
