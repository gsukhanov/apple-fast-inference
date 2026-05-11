#pragma once

#include "../../tensor.hpp"

namespace fastinf {

template<DType _IntDType, DType _FloatDType>
struct QuantizeDesc {
    using int_t = typename DTypeTraits<_IntDType>::type;
    using float_t = typename DTypeTraits<_FloatDType>::type;

    int_t zero_point = 0;
    float_t scale = 1;

    QuantizeDesc(int_t _zp, float_t _sc) : zero_point(_zp), scale(_sc) {
        static_assert(std::is_integral<int_t>, "Zero-point must be an integer.");
        static_assert(std::is_floating_point<float_t>, "Scale must be a real number.");
    }
};

using QTensor = Tensor<_IntDType, DeviceLikeType::neon>;
using RTensor = Tensor<_FloatDType, DeviceLikeType::neon>;
using LTensor = Tensor<_AccDType, DeviceLikeType::neon>;

using QPair = std::pair<QTensor, QuantizeDesc<_IntDType, _FloatDType>>;
using RPair = std::pair<RTensor, QuantizeDesc<_IntDType, _FloatDType>>;
using LPair = std::pair<LTensor, QuantizeDesc<_IntDType, _FloatDType>>;

template<DType _IntDType, DType _FloatDType>
QPair quantize(RTensor& input);

template<DType _IntDType, DType _FloatDType>
RTensor dequantize(QPair& input);

template <DType _AccDType, DType _IntDType, DType _FloatDType>
QPair rescale(LPair& input);
};

#include "../../../../../src/core/backend/neon/quantize.hpp.inl"