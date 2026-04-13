#include "fastinf/nn/ops/relu.hpp"

#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> relu(const TensorView<_DType, _Device>& input) {
    Tensor<_DType, _Device> output(input);
    for (auto& el : output) {
        el = std::max(el, typename Tensor<_DType, _Device>::scalar_t{});
    }
    return output;
}
}  // namespace nn
}  // namespace fastinf
