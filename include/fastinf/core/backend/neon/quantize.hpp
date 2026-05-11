#pragma once

#include <utility>

#include "../../tensor.hpp"

namespace fastinf {

template<DType _IntDType, DType _FloatDType>
struct QuantizeDesc {
    using int_t = typename DTypeTraits<_IntDType>::type;
    using float_t = typename DTypeTraits<_FloatDType>::type;

    int_t zero_point = 0;
    float_t scale = 1;

    QuantizeDesc(int_t _zp, float_t _sc) : zero_point(_zp), scale(_sc) {
        static_assert(std::is_integral<int_t>::value,
                      "Zero-point must be an integer.");
        static_assert(std::is_floating_point<float_t>::value,
                      "Scale must be a real number.");
    }
};

template<DType _IntDType>
using QTensor = Tensor<_IntDType, DeviceLikeType::neon>;

template<DType _FloatDType>
using RTensor = Tensor<_FloatDType, DeviceLikeType::neon>;

template<DType _AccDType>
using LTensor = Tensor<_AccDType, DeviceLikeType::neon>;

template<DType _IntDType, DType _FloatDType>
using QPair = std::pair<QTensor<_IntDType>, QuantizeDesc<_IntDType, _FloatDType>>;

template<DType _FloatDType, DType _IntDType>
using RPair = std::pair<RTensor<_FloatDType>, QuantizeDesc<_IntDType, _FloatDType>>;

template<DType _AccDType, DType _IntDType, DType _FloatDType>
using LPair = std::pair<LTensor<_AccDType>, QuantizeDesc<_IntDType, _FloatDType>>;

template<DType _IntDType, DType _FloatDType>
QPair<_IntDType, _FloatDType> quantize(RTensor<_FloatDType>& input);

template<DType _IntDType, DType _FloatDType>
RTensor<_FloatDType> dequantize(QPair<_IntDType, _FloatDType>& input);

template <DType _AccDType, DType _IntDType, DType _FloatDType>
QPair<_IntDType, _FloatDType> rescale(
    LPair<_AccDType, _IntDType, _FloatDType>& input);
};

#include "../../../../../src/core/backend/neon/quantize.hpp.inl"
