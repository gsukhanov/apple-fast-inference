#pragma once

#include "../device.hpp"
#include "../dtype.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class Tensor;

template <DType _DType, DeviceLikeType _Device>
class TensorView;

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> matmul(const TensorView<_DType, _Device>& lhs,
                               const TensorView<_DType, _Device>& rhs);
}  // namespace fastinf

#include "../../../../src/core/ops/matmul.hpp.inl"
