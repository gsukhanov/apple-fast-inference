#include <Accelerate/Accelerate.h>

#include "fastinf/core/tensor.hpp"
#include "fastinf/nn/backend/amx/linear.hpp"

namespace fastinf {
namespace nn {

template <>
inline Tensor<DType::float32, DeviceLikeType::amx> linear(
    const TensorView<DType::float32, DeviceLikeType::amx>& input,
    const TensorView<DType::float32, DeviceLikeType::amx>& weight,
    std::optional<std::reference_wrapper<
        const TensorView<DType::float32, DeviceLikeType::amx>>>
        bias) {
    using tensor_t = Tensor<DType::float32, DeviceLikeType::amx>;
    using scalar_t = typename tensor_t::scalar_t;

    if (input.dim() != 2) {
        throw std::invalid_argument("Linear expects 2D input [N, in_features]");
    }
    if (weight.dim() != 2) {
        throw std::invalid_argument(
            "Linear weight expects 2D shape [out_features, in_features]");
    }

    const auto batch = input.shape()[0];
    const auto in_features = input.shape()[1];
    const auto out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::invalid_argument("Input features do not match Linear layer");
    }
    if (bias.has_value()) {
        const auto& bias_view = bias->get();
        if (bias_view.dim() != 1 || bias_view.shape()[0] != out_features) {
            throw std::invalid_argument("Linear bias shape mismatch");
        }
    }

    tensor_t input_tmp, weight_tmp;
    scalar_t* input_data = backend::ensure_contiguous_data(input, input_tmp);
    scalar_t* weight_data = backend::ensure_contiguous_data(weight, weight_tmp);

    tensor_t output({batch, out_features}, scalar_t{});

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(batch), static_cast<int>(out_features),
                static_cast<int>(in_features), 1.0f, input_data,
                static_cast<int>(in_features), weight_data,
                static_cast<int>(in_features), 0.0f, output.data(),
                static_cast<int>(out_features));

    if (bias.has_value()) {
        for (std::int64_t n = 0; n < batch; ++n) {
            output.slice({n}) += bias->get();
        }
    }

    return output;
}

template <>
inline Tensor<DType::float64, DeviceLikeType::amx> linear(
    const TensorView<DType::float64, DeviceLikeType::amx>& input,
    const TensorView<DType::float64, DeviceLikeType::amx>& weight,
    std::optional<std::reference_wrapper<
        const TensorView<DType::float64, DeviceLikeType::amx>>>
        bias) {
    using tensor_t = Tensor<DType::float64, DeviceLikeType::amx>;
    using scalar_t = typename tensor_t::scalar_t;

    if (input.dim() != 2) {
        throw std::invalid_argument("Linear expects 2D input [N, in_features]");
    }
    if (weight.dim() != 2) {
        throw std::invalid_argument(
            "Linear weight expects 2D shape [out_features, in_features]");
    }

    const auto batch = input.shape()[0];
    const auto in_features = input.shape()[1];
    const auto out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::invalid_argument("Input features do not match Linear layer");
    }
    if (bias.has_value()) {
        const auto& bias_view = bias->get();
        if (bias_view.dim() != 1 || bias_view.shape()[0] != out_features) {
            throw std::invalid_argument("Linear bias shape mismatch");
        }
    }

    tensor_t input_tmp, weight_tmp;
    scalar_t* input_data = backend::ensure_contiguous_data(input, input_tmp);
    scalar_t* weight_data = backend::ensure_contiguous_data(weight, weight_tmp);

    tensor_t output({batch, out_features}, scalar_t{});

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(batch), static_cast<int>(out_features),
                static_cast<int>(in_features), 1.0, input_data,
                static_cast<int>(in_features), weight_data,
                static_cast<int>(in_features), 0.0, output.data(),
                static_cast<int>(out_features));

    if (bias.has_value()) {
        for (std::int64_t n = 0; n < batch; ++n) {
            output.slice({n}) += bias->get();
        }
    }

    return output;
}

}  // namespace nn
}  // namespace fastinf
