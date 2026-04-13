#include "fastinf/core/ops/common.hpp"
#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace {
template <DType _DType, DeviceLikeType _Device, typename TView, typename Op>
Tensor<_DType, _Device> unary_scalar_op(
    const TView& input, typename Tensor<_DType, _Device>::scalar_t scalar,
    Op&& op) {
    Tensor<_DType, _Device> result(
        input.shape(), typename Tensor<_DType, _Device>::scalar_t{});
    auto it_out = result.begin();
    for (auto it_in = input.begin(); it_in != input.end(); ++it_in, ++it_out) {
        *it_out = op(*it_in, scalar);
    }
    return result;
}

template <DType _DType, DeviceLikeType _Device, typename TLhs, typename TRhs,
          typename Op>
Tensor<_DType, _Device> binary_elementwise_op(const TLhs& lhs, const TRhs& rhs,
                                              Op&& op) {
    check_same_shape(lhs, rhs);
    Tensor<_DType, _Device> result(
        lhs.shape(), typename Tensor<_DType, _Device>::scalar_t{});
    auto it_out = result.begin();
    auto it_rhs = rhs.begin();
    for (auto it_lhs = lhs.begin(); it_lhs != lhs.end();
         ++it_lhs, ++it_rhs, ++it_out) {
        *it_out = op(*it_lhs, *it_rhs);
    }
    return result;
}
}  // namespace

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> add(const TensorView<_DType, _Device>& lhs,
                            const TensorView<_DType, _Device>& rhs) {
    return binary_elementwise_op<_DType, _Device>(
        lhs, rhs, [](const auto& a, const auto& b) { return a + b; });
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> add(const TensorView<_DType, _Device>& input,
                            typename Tensor<_DType, _Device>::scalar_t scalar) {
    return unary_scalar_op<_DType, _Device>(
        input, scalar, [](const auto& a, const auto& b) { return a + b; });
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> sub(const TensorView<_DType, _Device>& lhs,
                            const TensorView<_DType, _Device>& rhs) {
    return binary_elementwise_op<_DType, _Device>(
        lhs, rhs, [](const auto& a, const auto& b) { return a - b; });
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> sub(const TensorView<_DType, _Device>& input,
                            typename Tensor<_DType, _Device>::scalar_t scalar) {
    return unary_scalar_op<_DType, _Device>(
        input, scalar, [](const auto& a, const auto& b) { return a - b; });
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> multiply(const TensorView<_DType, _Device>& lhs,
                                 const TensorView<_DType, _Device>& rhs) {
    return binary_elementwise_op<_DType, _Device>(
        lhs, rhs, [](const auto& a, const auto& b) { return a * b; });
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> multiply(
    const TensorView<_DType, _Device>& input,
    typename Tensor<_DType, _Device>::scalar_t scalar) {
    return unary_scalar_op<_DType, _Device>(
        input, scalar, [](const auto& a, const auto& b) { return a * b; });
}
}  // namespace fastinf
