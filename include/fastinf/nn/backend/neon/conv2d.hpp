#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "layer.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu> // maybe new device or version instead of device?
class ConvIm2Col : public Layer<_DType, _Device> {
public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;
    using layer_state_t = typename Layer<_DType, _Device>::layer_state_t;
    struct size_2_t {
        std::int64_t h, w;
    };

    ConvIm2Col(std::int64_t in_channels, std::int64_t out_channels, size_2_t kernel,
           size_2_t stride = {1, 1}, size_2_t padding = {0, 0},
           size_2_t dilation = {1, 1}, std::int64_t groups = 1,
           bool bias = true);

    ConvIm2Col(std::int64_t in_channels, std::int64_t out_channels,
           std::int64_t kernel, std::int64_t stride = 1,
           std::int64_t padding = 0, std::int64_t dilation = 1,
           std::int64_t groups = 1, bool bias = true);

    std::string name() const override;
    tensor_t forward(const tensor_t& input) const override;
    bool load_state(const layer_state_t& state) override;

    const tensor_t& weight() const;
    const tensor_t& bias() const;

private:
    std::int64_t in_channels_;
    std::int64_t out_channels_;
    size_2_t kernel_;
    size_2_t stride_;
    size_2_t padding_;
    size_2_t dilation_;
    std::int64_t groups_;
    bool use_bias_;

    tensor_t weight_;
    tensor_t bias_;

    std::int64_t output_height(std::int64_t input_h) const;
    std::int64_t output_width(std::int64_t input_w) const;
};
} // namespace nn
} // namespace fastinf

#include "../../../src/nn/backend/neon/conv2d.hpp.inl"