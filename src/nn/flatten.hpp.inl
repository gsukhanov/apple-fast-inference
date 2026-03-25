#include "fastinf/nn/flatten.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
std::string Flatten<_DType, _Device>::name() const {
    return "nn.Flatten";
}

template <DType _DType, DeviceLikeType _Device>
typename Flatten<_DType, _Device>::tensor_t Flatten<_DType, _Device>::forward(
    const tensor_t& input) const {
    if (input.dim() == 0) {
        throw std::invalid_argument("Flatten expects tensor rank >= 1");
    }

    const auto& shape = input.shape();
    const std::int64_t batch = shape[0];

    std::int64_t features = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        features *= shape[i];
    }

    tensor_t output({batch, features}, scalar_t{});
    auto out_it = output.begin();
    for (auto in_it = input.begin(); in_it != input.end(); ++in_it, ++out_it) {
        *out_it = *in_it;
    }
    return output;
}

}  // namespace nn
}  // namespace fastinf
