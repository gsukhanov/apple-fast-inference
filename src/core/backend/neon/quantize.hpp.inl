#include "fastinf/core/backend/neon/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fastinf {

template<DType _IntDType, DType _FloatDType>
QPair<_IntDType, _FloatDType> quantize(RTensor<_FloatDType>& input) {
    using int_t = typename DTypeTraits<_IntDType>::type;
    using float_t = typename DTypeTraits<_FloatDType>::type;

    float_t min_value = float_t{0};
    float_t max_value = float_t{0};
    for (auto it = input.begin(); it != input.end(); ++it) {
        min_value = std::min(min_value, *it);
        max_value = std::max(max_value, *it);
    }

    const auto int_max = static_cast<float_t>(DTypeTraits<_IntDType>::max);
    const float_t abs_max = std::max(std::abs(min_value), std::abs(max_value));
    const float_t scale = abs_max == float_t{0} ? float_t{1} : abs_max / int_max;
    const int_t zero_point = int_t{0};

    QTensor<_IntDType> result(input.shape(), int_t{0});
    auto it1 = input.begin();
    auto it2 = result.begin();
    for (; it1 != input.end(); ++it1, ++it2) {
        const auto quantized = std::llround(*it1 / scale) + zero_point;
        const auto clamped = std::clamp<long long>(
            quantized,
            static_cast<long long>(std::numeric_limits<int_t>::lowest()),
            static_cast<long long>(DTypeTraits<_IntDType>::max));
        *it2 = static_cast<int_t>(clamped);
    }

    result.set_quantization(
        TensorQuantization{_FloatDType, _IntDType, zero_point, scale});
    return std::make_pair(result,
                          QuantizeDesc<_IntDType, _FloatDType>(zero_point,
                                                                scale));
}

template<DType _IntDType, DType _FloatDType>
RTensor<_FloatDType> dequantize(QPair<_IntDType, _FloatDType>& input) {
    using float_t = typename DTypeTraits<_FloatDType>::type;

    const auto zero_point = input.second.zero_point;
    const auto scale = input.second.scale;

    RTensor<_FloatDType> result(input.first.shape(), float_t{0});
    auto it1 = input.first.begin();
    auto it2 = result.begin();
    for (; it1 != input.first.end(); ++it1, ++it2) {
        *it2 = static_cast<float_t>(*it1 - zero_point) * scale;
    }

    return result;
}

template <DType _AccDType, DType _IntDType, DType _FloatDType>
QPair<_IntDType, _FloatDType> rescale(
    LPair<_AccDType, _IntDType, _FloatDType>& input) {
    using int_t = typename DTypeTraits<_IntDType>::type;
    using float_t = typename DTypeTraits<_FloatDType>::type;

    const auto zero_point = input.second.zero_point;
    const auto scale = input.second.scale;

    QTensor<_IntDType> result(input.first.shape(), int_t{0});
    auto it1 = input.first.begin();
    auto it2 = result.begin();
    for (; it1 != input.first.end(); ++it1, ++it2) {
        const auto scaled =
            std::llround(static_cast<float_t>(*it1 - zero_point) * scale);
        const auto clamped = std::clamp<long long>(
            scaled,
            static_cast<long long>(std::numeric_limits<int_t>::lowest()),
            static_cast<long long>(DTypeTraits<_IntDType>::max));
        *it2 = static_cast<int_t>(clamped);
    }

    result.set_quantization(
        TensorQuantization{_AccDType, _IntDType, 0, static_cast<double>(scale)});
    return std::make_pair(result,
                          QuantizeDesc<_IntDType, _FloatDType>(0, scale));
}

}  // namespace fastinf
