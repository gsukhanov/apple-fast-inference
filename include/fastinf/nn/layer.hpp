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
    Layer() = default;
    Layer(const Layer&) = default;
    Layer& operator=(const Layer&) = default;
    virtual ~Layer() = default;

    virtual std::string name() const = 0;
    virtual void print(std::ostream& o, int level = 0) const;
    virtual Tensor<_DType, _Device> forward(
        const Tensor<_DType, _Device>& input) const = 0;
};

template <DType _DType, DeviceLikeType _Device>
std::ostream& operator<<(std::ostream& o, const Layer<_DType, _Device>& m);

};  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/layer.hpp.inl"
