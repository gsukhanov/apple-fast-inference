#include "fastinf/nn/softmax.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Softmax<_DType, _Device>::Softmax(std::optional<int> dim) : dim_(dim) {
}

template <DType _DType, DeviceLikeType _Device>
std::string Softmax<_DType, _Device>::name() const {
    return "nn.Softmax";
}

template <DType _DType, DeviceLikeType _Device>
typename Softmax<_DType, _Device>::tensor_t Softmax<_DType, _Device>::forward(
    const tensor_t& input) const {
    int axis = dim_.value_or(input.dim() - 1);
    if (axis < 0) {
        axis += input.dim();
    }
    if (axis < 0 || axis >= input.dim()) {
        throw std::invalid_argument("Softmax axis out of range");
    }
    return softmax<_DType, _Device>(input.view(), axis);
}

}  // namespace nn
}  // namespace fastinf
