#include "fastinf/nn/lenet.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
LeNet<_DType, _Device>::LeNet(std::int64_t num_classes)
    : conv1(1, 6, 5),
      relu1(),
      pool1(2, 2),
      conv2(6, 16, 5),
      relu2(),
      pool2(2, 2),
      flatten(),
      fc1(16 * 4 * 4, 120),
      relu3(),
      fc2(120, 84),
      relu4(),
      fc3(84, num_classes) {
    conv1.bind_identity("conv1");
    conv2.bind_identity("conv2");
    fc1.bind_identity("fc1");
    fc2.bind_identity("fc2");
    fc3.bind_identity("fc3");
    this->bind_children(conv1, conv2, fc1, fc2, fc3);
}

template <DType _DType, DeviceLikeType _Device>
std::string LeNet<_DType, _Device>::name() const {
    return "nn.LeNet";
}

template <DType _DType, DeviceLikeType _Device>
typename LeNet<_DType, _Device>::tensor_t LeNet<_DType, _Device>::forward(
    const tensor_t& input) const {
    tensor_t output = input;
    output = pool1(relu1(conv1(output)));
    output = pool2(relu2(conv2(output)));

    output = flatten(output);
    output = relu3(fc1(output));
    output = relu4(fc2(output));
    output = fc3(output);

    return output;
}

}  // namespace nn
}  // namespace fastinf
