#pragma once

#include "../../tensor.hpp"

#define KERNEL_WIDTH 3
#define KERNEL_DEPTH 4
#define L1_BLOCK_WIDTH 48
#define L1_BlOCK_DEPTH 16
#define L2_BLOCK_WIDTH 64
#define L2_BLOCK_DEPTH 128

namespace fastinf {
    //todo: make this a template class with different shapes of blocks

    template<DType _DType, DeviceLikeType _Device>
    static Tensor<_DType, _Device> pack_input(Tensor<_DType, _Device>& input);

    template<DType _DType, DeviceLikeType _Device>
    static Tensor<_DType, _Device> pack_result(Tensor<_DType, _Device>& input);

    template<DType _DType, DeviceLikeType _Device>
    static Tensor<_DType, _Device> unpack_result(Tensor<_DType, _Device>& input);
}

#include "../../../../../src/core/backend/neon/blocking.hpp.inl"