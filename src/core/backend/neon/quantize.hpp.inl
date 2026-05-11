#include "fastinf/core/backend/neon/quantize.hpp"

namespace fastinf {

template<DType _IntDType, DType _FloatDType>
QPair quantize(RTensor& input) {

    using int_t = typename DTypeTraits<_IntDType>::type;
    using float_t = typename DTypeTraits<_floatDType>::type;


    float_t min(0), max(0);

    for (auto it = input.begin(); it != input.end(); ++it) {
        min = std::min(min, *it);
        max = std::max(max, *it);
    }

    int_t int_max = DTypeTraits<_IntDType>::max;

    int_t zero_point(0);
    float_t scale(std::max(std::abs(min), std::abs(max)) / int_max);

    QTensor result(input.shape(), 0);

    for (auto it1 = input.begin(), it2 = result.begin(); it1 < input.end(); ++it1, ++it2) {
        *it2 = (int_t)(*it1 / scale) + zero_point;
    }

    return std::make_pair(QTensor(), QuantizeDesc(zero_point, scale));
}

template<DType _IntDType, DType _FloatDType>
RTensor dequantize(QPair& input) {

    using int_t = typename DTypeTraits<_IntDType>::type;
    using float_t = typename DTypeTraits<_floatDType>::type;

    int_t zero_point = input.second.zero_point;
    float_t scale = input.second.scale;

    RTensor result(input.first.shape(), 0);

    for (auto it1 = input.first.begin(), it2 = result.begin(); it1 < input.first.end(); ++it1, ++it2) {
        *it2 = (*it1 - zero_point) * scale;
    }

    return result;
}

template <DType _AccDType, DType _IntDType, DType _FloatDType>
QPair rescale(LPair& input) {

    using long_t = typename DTypeTraits<_AccDType>::type;
    using int_t = typename DTypeTraits<_IntDType>::type;
    using float_t = typename DTypeTraits<_floatDType>::type;

    std::size_t int_size = DTypeTraits<_IntDType>::size;
    long_t long_max = DTypeTraits<_AccDType>::max;

    QTensor result(input.first.shape(), 0);

    float_t scale = input.second.scale;
    long_t zero_point = input.second.zero_point;
    
    for (auto it1 = input.first.begin(), it2 = result.begin(); it1 < input.first.end(); ++it1, ++it2) {
        *it2 = (int_t)((*it1 - zero_point) >> (2 >> (3 * (int_size + 1))));
    }

    return std::make_pair(QTensor, QuantizeDesc(0, scale * (long_max >> (2 >> (3 * int_size)))));
}

};

// r = s * (i - z)
// z = i - r / s
// r1 * r2 = s1 * s2 * (i - z) * (i - z)