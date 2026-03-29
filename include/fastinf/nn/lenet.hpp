#pragma once

#include "avg_pool2d.hpp"
#include "conv2d.hpp"
#include "flatten.hpp"
#include "layer.hpp"
#include "linear.hpp"
#include "relu.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class LeNet : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    LeNet(std::int64_t num_classes = 10);

    std::string name() const override;

    tensor_t forward(const tensor_t& input) const override;

 private:
    Conv2d<_DType, _Device> conv1, conv2;
    ReLU<_DType, _Device> relu1, relu2, relu3, relu4;
    AvgPool2d<_DType, _Device> pool1, pool2;
    Flatten<_DType, _Device> flatten;
    Linear<_DType, _Device> fc1, fc2, fc3;
};
}  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/lenet.hpp.inl"
