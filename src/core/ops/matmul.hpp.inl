#include "fastinf/core/tensor.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> matmul(const TensorView<_DType, _Device>& lhs,
                               const TensorView<_DType, _Device>& rhs) {
    using scalar_t = typename Tensor<_DType, _Device>::scalar_t;

    if (lhs.dim() == 1 && rhs.dim() == 1) {
        if (lhs.shape()[0] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension 0");
        }
        scalar_t acc{};
        for (std::int64_t i = 0; i < lhs.shape()[0]; ++i) {
            acc += lhs.at({i}) * rhs.at({i});
        }

        return Tensor<_DType, _Device>({}, std::vector<scalar_t>{acc});
    }

    if (lhs.dim() == 2 && rhs.dim() == 1) {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        Tensor<_DType, _Device> result({lhs.shape()[0]}, scalar_t{});
        for (std::int64_t i = 0; i < lhs.shape()[0]; ++i) {
            scalar_t acc{};
            for (std::int64_t k = 0; k < lhs.shape()[1]; ++k) {
                acc += lhs.at({i, k}) * rhs.at({k});
            }
            result.at({i}) = acc;
        }

        return result;
    }

    if (lhs.dim() == 1 && rhs.dim() == 2) {
        if (lhs.shape()[0] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }
        Tensor<_DType, _Device> result({rhs.shape()[1]}, scalar_t{});
        for (std::int64_t j = 0; j < rhs.shape()[1]; ++j) {
            scalar_t acc{};
            for (std::int64_t i = 0; i < lhs.shape()[0]; ++i) {
                acc += lhs.at({i}) * rhs.at({i, j});
            }
            result.at({j}) = acc;
        }

        return result;
    }

    if (lhs.dim() == 2 && rhs.dim() == 2) {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        Tensor<_DType, _Device> result({lhs.shape()[0], rhs.shape()[1]},
                                       scalar_t{});
        for (std::int64_t i = 0; i < lhs.shape()[0]; ++i) {
            for (std::int64_t j = 0; j < rhs.shape()[1]; ++j) {
                scalar_t acc{};
                for (std::int64_t k = 0; k < lhs.shape()[1]; ++k) {
                    acc += lhs.at({i, k}) * rhs.at({k, j});
                }
                result.at({i, j}) = acc;
            }
        }

        return result;
    }

    throw std::runtime_error(
        "The size of tensor a and tensor b must be less than 3");
}
}  // namespace fastinf
