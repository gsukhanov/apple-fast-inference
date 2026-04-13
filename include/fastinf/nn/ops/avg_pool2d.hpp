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
Tensor<_DType, _Device> avg_pool2d(const TensorView<_DType, _Device>& input,
                                   std::int64_t kernel_h,
                                   std::int64_t kernel_w,
                                   std::int64_t stride_h,
                                   std::int64_t stride_w,
                                   std::int64_t padding_h,
                                   std::int64_t padding_w);
}  // namespace nn
}  // namespace fastinf

#include "../../../../src/nn/ops/avg_pool2d.hpp.inl"
