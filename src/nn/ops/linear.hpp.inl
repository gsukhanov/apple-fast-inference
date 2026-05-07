#include "fastinf/nn/ops/linear.hpp"

#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> linear(const TensorView<_DType, _Device>& input,
                               const TensorView<_DType, _Device>& weight,
                               std::optional<std::reference_wrapper<
                                   const TensorView<_DType, _Device>>> bias) {
    if (input.dim() != 2) {
        throw std::invalid_argument("Linear expects 2D input [N, in_features]");
    }
    if (weight.dim() != 2) {
        throw std::invalid_argument(
            "Linear weight expects 2D shape [out_features, in_features]");
    }

    const auto& input_shape = input.shape();
    const auto& weight_shape = weight.shape();
    const auto batch = input_shape[0];
    const auto in_features = input_shape[1];
    const auto out_features = weight_shape[0];

    if (weight_shape[1] != in_features) {
        throw std::invalid_argument("Input features do not match Linear layer");
    }

    if (bias.has_value()) {
        const auto& bias_view = bias->get();
        if (bias_view.dim() != 1 || bias_view.shape()[0] != out_features) {
            throw std::invalid_argument("Linear bias shape mismatch");
        }
    }

    Tensor<_DType, _Device> output({batch, out_features},
                                   typename Tensor<_DType, _Device>::scalar_t{});

    for (std::int64_t n = 0; n < batch; ++n) {
        output.slice({n}) = input.slice({n}).mul(weight.t());
        if (bias.has_value()) {
            output.slice({n}) += bias->get();
        }
    }

    return output;
}
}  // namespace nn
}  // namespace fastinf
