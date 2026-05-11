#pragma once

#include "../../device.hpp"
#include "../../dtype.hpp"

#include "quantize.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class Tensor;

template <DType _DType, DeviceLikeType _Device>
class TensorView;

template <DType _DType, DType _FloatDType>
Tensor<_DType, DeviceLikeType::neon> matmul(const TensorView<_DType, DeviceLikeType::neon>& lhs,
                               const TensorView<_DType, DeviceLikeType::neon>& rhs,
                               const QuantizeDesc<_DType, _FloatDType> lhs_qdesc = {0, 1 / DTypeTraits<_DType>::max},
                               const QuantizeDesc<_DType, _FloatDType> rhs_qdesc = {0, 1 / DTypeTraits<_DType>::max});
}  // namespace fastinf


#include "../../../../../src/core/backend/neon/matmul.hpp.inl"
