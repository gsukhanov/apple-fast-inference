#include "fastinf/nn/ops/flatten.hpp"

#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> flatten(const TensorView<_DType, _Device>& input) {
    if (input.dim() == 0) {
        throw std::invalid_argument("Flatten expects tensor rank >= 1");
    }

    const auto& shape = input.shape();
    const std::int64_t batch = shape[0];

    std::int64_t features = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        features *= shape[i];
    }

    Tensor<_DType, _Device> output({batch, features},
                                   typename Tensor<_DType, _Device>::scalar_t{});
    auto out_it = output.begin();
    for (auto in_it = input.begin(); in_it != input.end(); ++in_it, ++out_it) {
        *out_it = *in_it;
    }
    return output;
}
}  // namespace nn
}  // namespace fastinf
