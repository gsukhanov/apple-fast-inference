#include "fastinf/nn/relu.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
std::string ReLU<_DType, _Device>::name() const {
    return "nn.ReLU";
}

template <DType _DType, DeviceLikeType _Device>
typename ReLU<_DType, _Device>::tensor_t ReLU<_DType, _Device>::forward(
    const tensor_t& input) const {
    return nn::relu<_DType, _Device>(input.view());
}

}  // namespace nn
}  // namespace fastinf
