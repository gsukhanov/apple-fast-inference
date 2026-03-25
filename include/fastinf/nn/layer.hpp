#pragma once

#include <ostream>
#include <string>

#include "fastinf/core/device.hpp"
#include "fastinf/core/dtype.hpp"
#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class Layer {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    Layer() = default;
    Layer(const Layer&) = default;
    Layer& operator=(const Layer&) = default;
    virtual ~Layer() = default;

    virtual std::string name() const = 0;
    virtual void print(std::ostream& o, int level = 0) const;
    virtual tensor_t forward(const tensor_t& input) const = 0;
};

template <DType _DType, DeviceLikeType _Device>
std::ostream& operator<<(std::ostream& o, const Layer<_DType, _Device>& m);

};  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/layer.hpp.inl"
