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
    return nn::flatten<_DType, _Device>(input.view());
}

}  // namespace nn
}  // namespace fastinf
