#pragma once

#include "../../tensor.hpp"

#define KERNEL_WIDTH 3
#define KERNEL_DEPTH 4
#define KERNEL_RESULT_WIDTH 9
#define L1_BLOCK_WIDTH 48
#define L1_BlOCK_DEPTH 16
#define L1_BLOCK_DEPTH L1_BlOCK_DEPTH
#define L2_BLOCK_WIDTH 64
#define L2_BLOCK_DEPTH 128

namespace fastinf {
    enum class BlockingRole { lhs, rhs, result };

    template<DType _DType, DeviceLikeType _Device>
    Tensor<_DType, _Device> pack_input(const Tensor<_DType, _Device>& input);

    template<DType _DType, DeviceLikeType _Device>
    Tensor<_DType, _Device> pack_input(const TensorView<_DType, _Device>& input);

    template<DType _DType, DeviceLikeType _Device>
    Tensor<_DType, _Device> pack_result(const Tensor<_DType, _Device>& input);

    template<DType _DType, DeviceLikeType _Device>
    Tensor<_DType, _Device> pack_result(const TensorView<_DType, _Device>& input);

    template<DType _DType, DeviceLikeType _Device>
    Tensor<_DType, _Device> unpack_result(const Tensor<_DType, _Device>& input,
                                          const Shape& output_shape);
}

#include "../../../../../src/core/backend/neon/blocking.hpp.inl"
