#pragma once

#include "../device.hpp"
#include "../dtype.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class Tensor;

template <DType _DType, DeviceLikeType _Device>
class TensorView;

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> add(const TensorView<_DType, _Device>& lhs,
                            const TensorView<_DType, _Device>& rhs);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> add(const TensorView<_DType, _Device>& input,
                            typename Tensor<_DType, _Device>::scalar_t scalar);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> sub(const TensorView<_DType, _Device>& lhs,
                            const TensorView<_DType, _Device>& rhs);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> sub(const TensorView<_DType, _Device>& input,
                            typename Tensor<_DType, _Device>::scalar_t scalar);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> multiply(const TensorView<_DType, _Device>& lhs,
                                 const TensorView<_DType, _Device>& rhs);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> multiply(
    const TensorView<_DType, _Device>& input,
    typename Tensor<_DType, _Device>::scalar_t scalar);
}  // namespace fastinf

#include "../../../../src/core/ops/elementwise.hpp.inl"
