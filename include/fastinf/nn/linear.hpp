#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "layer.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class Linear : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    Linear(std::int64_t in_features, std::int64_t out_features,
           bool bias = true);

    std::string name() const override;
    tensor_t forward(const tensor_t& input) const override;

    const tensor_t& weight() const;
    const tensor_t& bias() const;

    void load_weights(const tensor_t& weight, const tensor_t& bias);

 private:
    std::int64_t in_features_;
    std::int64_t out_features_;
    bool use_bias_;

    tensor_t weight_;
    tensor_t bias_;
};
}  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/linear.hpp.inl"
