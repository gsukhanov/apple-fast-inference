#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "layer.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class Softmax : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    explicit Softmax(std::optional<int> dim = std::nullopt);

    std::string name() const override;
    tensor_t forward(const tensor_t& input) const override;

 private:
    std::optional<int> dim_;
};
}  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/softmax.hpp.inl"
