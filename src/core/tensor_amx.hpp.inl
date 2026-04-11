#include <Accelerate/Accelerate.h>

#include "fastinf/core/tensor.hpp"

namespace fastinf {
// template <>
// Tensor<DType::int8, DeviceLikeType::amx>
// Tensor<DType::int8, DeviceLikeType::amx>::mul(const Tensor& other) const {
// }

// template <>
// Tensor<DType::int16, DeviceLikeType::amx>
// Tensor<DType::int16, DeviceLikeType::amx>::mul(const Tensor& other) const {
// }

// template <>
// Tensor<DType::int32, DeviceLikeType::amx>
// Tensor<DType::int32, DeviceLikeType::amx>::mul(const Tensor& other) const {
// }

// template <>
// Tensor<DType::int64, DeviceLikeType::amx>
// Tensor<DType::int64, DeviceLikeType::amx>::mul(const Tensor& other) const {
// }

template <>
inline Tensor<DType::float32, DeviceLikeType::amx>
Tensor<DType::float32, DeviceLikeType::amx>::mul(const Tensor& other) const {
    Tensor a, b;
    const scalar_t *A = data_ + desc_.offset_,
                   *B = other.data_ + other.desc_.offset_;

    if (!is_contiguous()) {
        a = contiguous();
        A = a.data_;
    }

    if (!other.is_contiguous()) {
        b = other.contiguous();
        B = b.data_;
    }

    if (dim() == 1 && other.dim() == 1) {
        if (desc_.shape_[0] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension 0");
        }

        const int N = desc_.shape_[0];
        scalar_t acc = cblas_sdot(N, A, 1, B, 1);

        return Tensor({}, std::vector<scalar_t>{acc});
    }

    if (dim() == 2 && other.dim() == 1) {
        if (desc_.shape_[1] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension");
        }

        Tensor result({desc_.shape_[0]}, scalar_t{});
        const int M = desc_.shape_[0];
        const int N = desc_.shape_[1];

        cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A, N, B, 1, 0.0f,
                    result.data_, 1);

        return result;
    }

    if (dim() == 1 && other.dim() == 2) {
        if (desc_.shape_[0] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension");
        }

        Tensor result({other.desc_.shape_[1]}, scalar_t{});
        const int M = other.desc_.shape_[0];
        const int N = other.desc_.shape_[1];

        cblas_sgemv(CblasRowMajor, CblasTrans, M, N, 1.0f, B, N, A, 1, 0.0f,
                    result.data_, 1);

        return result;
    }

    if (dim() == 2 && other.dim() == 2) {
        if (desc_.shape_[1] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension");
        }

        Tensor result({desc_.shape_[0], other.desc_.shape_[1]}, scalar_t{});
        const int M = desc_.shape_[0];
        const int N = other.desc_.shape_[1];
        const int K = desc_.shape_[1];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A,
                    K, B, N, 0.0f, result.data_, N);

        return result;
    }

    throw std::runtime_error(
        "The size of tensor a and tensor b must be less than 3");
}

template <>
inline Tensor<DType::float64, DeviceLikeType::amx>
Tensor<DType::float64, DeviceLikeType::amx>::mul(const Tensor& other) const {
    Tensor a, b;
    const scalar_t *A = data_ + desc_.offset_,
                   *B = other.data_ + other.desc_.offset_;

    if (!is_contiguous()) {
        a = contiguous();
        A = a.data_;
    }

    if (!other.is_contiguous()) {
        b = other.contiguous();
        B = b.data_;
    }

    if (dim() == 1 && other.dim() == 1) {
        if (desc_.shape_[0] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension 0");
        }

        const int N = desc_.shape_[0];
        scalar_t acc = cblas_ddot(N, A, 1, B, 1);

        return Tensor({}, std::vector<scalar_t>{acc});
    }

    if (dim() == 2 && other.dim() == 1) {
        if (desc_.shape_[1] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension");
        }

        Tensor result({desc_.shape_[0]}, scalar_t{});
        const int M = desc_.shape_[0];
        const int N = desc_.shape_[1];

        cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A, N, B, 1, 0.0f,
                    result.data_, 1);

        return result;
    }

    if (dim() == 1 && other.dim() == 2) {
        if (desc_.shape_[0] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension");
        }

        Tensor result({other.desc_.shape_[1]}, scalar_t{});
        const int M = other.desc_.shape_[0];
        const int N = other.desc_.shape_[1];

        cblas_dgemv(CblasRowMajor, CblasTrans, M, N, 1.0f, B, N, A, 1, 0.0f,
                    result.data_, 1);

        return result;
    }

    if (dim() == 2 && other.dim() == 2) {
        if (desc_.shape_[1] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b "
                "at "
                "non-singleton dimension");
        }

        Tensor result({desc_.shape_[0], other.desc_.shape_[1]}, scalar_t{});
        const int M = desc_.shape_[0];
        const int N = other.desc_.shape_[1];
        const int K = desc_.shape_[1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A,
                    K, B, N, 0.0f, result.data_, N);

        return result;
    }

    throw std::runtime_error(
        "The size of tensor a and tensor b must be less than 3");
}
}  // namespace fastinf
