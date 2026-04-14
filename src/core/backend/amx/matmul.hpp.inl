#include <Accelerate/Accelerate.h>

#include "fastinf/core/backend/common.hpp"
#include "fastinf/core/tensor.hpp"

namespace fastinf {

template <>
inline Tensor<DType::float32, DeviceLikeType::amx> matmul(
    const TensorView<DType::float32, DeviceLikeType::amx>& lhs,
    const TensorView<DType::float32, DeviceLikeType::amx>& rhs) {
    using tensor_t = Tensor<DType::float32, DeviceLikeType::amx>;
    using scalar_t = typename tensor_t::scalar_t;

    tensor_t lhs_tmp, rhs_tmp;
    scalar_t* lhs_data = backend::ensure_contiguous_data(lhs, lhs_tmp);
    scalar_t* rhs_data = backend::ensure_contiguous_data(rhs, rhs_tmp);

    if (lhs.dim() == 1 && rhs.dim() == 1) {
        if (lhs.shape()[0] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension 0");
        }

        const int n = static_cast<int>(lhs.shape()[0]);
        const scalar_t acc = cblas_sdot(n, lhs_data, 1, rhs_data, 1);
        return tensor_t({}, std::vector<scalar_t>{acc});
    }

    if (lhs.dim() == 2 && rhs.dim() == 1) {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        tensor_t result({lhs.shape()[0]}, scalar_t{});
        const int m = static_cast<int>(lhs.shape()[0]);
        const int n = static_cast<int>(lhs.shape()[1]);

        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, lhs_data, n,
                    rhs_data, 1, 0.0f, result.data(), 1);
        return result;
    }

    if (lhs.dim() == 1 && rhs.dim() == 2) {
        if (lhs.shape()[0] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        tensor_t result({rhs.shape()[1]}, scalar_t{});
        const int m = static_cast<int>(rhs.shape()[0]);
        const int n = static_cast<int>(rhs.shape()[1]);

        cblas_sgemv(CblasRowMajor, CblasTrans, m, n, 1.0f, rhs_data, n,
                    lhs_data, 1, 0.0f, result.data(), 1);
        return result;
    }

    if (lhs.dim() == 2 && rhs.dim() == 2) {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        tensor_t result({lhs.shape()[0], rhs.shape()[1]}, scalar_t{});
        const int m = static_cast<int>(lhs.shape()[0]);
        const int n = static_cast<int>(rhs.shape()[1]);
        const int k = static_cast<int>(lhs.shape()[1]);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                    lhs_data, k, rhs_data, n, 0.0f, result.data(), n);
        return result;
    }

    throw std::runtime_error(
        "The size of tensor a and tensor b must be less than 3");
}

template <>
inline Tensor<DType::float64, DeviceLikeType::amx> matmul(
    const TensorView<DType::float64, DeviceLikeType::amx>& lhs,
    const TensorView<DType::float64, DeviceLikeType::amx>& rhs) {
    using tensor_t = Tensor<DType::float64, DeviceLikeType::amx>;
    using scalar_t = typename tensor_t::scalar_t;

    tensor_t lhs_tmp, rhs_tmp;
    scalar_t* lhs_data = backend::ensure_contiguous_data(lhs, lhs_tmp);
    scalar_t* rhs_data = backend::ensure_contiguous_data(rhs, rhs_tmp);

    if (lhs.dim() == 1 && rhs.dim() == 1) {
        if (lhs.shape()[0] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension 0");
        }

        const int n = static_cast<int>(lhs.shape()[0]);
        const scalar_t acc = cblas_ddot(n, lhs_data, 1, rhs_data, 1);
        return tensor_t({}, std::vector<scalar_t>{acc});
    }

    if (lhs.dim() == 2 && rhs.dim() == 1) {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        tensor_t result({lhs.shape()[0]}, scalar_t{});
        const int m = static_cast<int>(lhs.shape()[0]);
        const int n = static_cast<int>(lhs.shape()[1]);

        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, lhs_data, n,
                    rhs_data, 1, 0.0, result.data(), 1);
        return result;
    }

    if (lhs.dim() == 1 && rhs.dim() == 2) {
        if (lhs.shape()[0] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        tensor_t result({rhs.shape()[1]}, scalar_t{});
        const int m = static_cast<int>(rhs.shape()[0]);
        const int n = static_cast<int>(rhs.shape()[1]);

        cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1.0, rhs_data, n, lhs_data,
                    1, 0.0, result.data(), 1);
        return result;
    }

    if (lhs.dim() == 2 && rhs.dim() == 2) {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at non-singleton dimension");
        }

        tensor_t result({lhs.shape()[0], rhs.shape()[1]}, scalar_t{});
        const int m = static_cast<int>(lhs.shape()[0]);
        const int n = static_cast<int>(rhs.shape()[1]);
        const int k = static_cast<int>(lhs.shape()[1]);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                    lhs_data, k, rhs_data, n, 0.0, result.data(), n);
        return result;
    }

    throw std::runtime_error(
        "The size of tensor a and tensor b must be less than 3");
}

}  // namespace fastinf
