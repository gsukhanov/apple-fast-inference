#pragma once

#include <algorithm>

#include "layer.hpp"
#include "ops.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class ReLU : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    ReLU() = default;

    std::string name() const override;
    tensor_t forward(const tensor_t& input) const override;
};
}  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/relu.hpp.inl"
#if FASTINF_HAS_ACCELERATE
#include "backend/amx/relu.hpp"
#endif
