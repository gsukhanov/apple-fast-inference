#include "fastinf/nn/conv2d.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Conv2d<_DType, _Device>::Conv2d(std::int64_t in_channels,
                                std::int64_t out_channels, size_2_t kernel,
                                size_2_t stride, size_2_t padding,
                                size_2_t dilation, std::int64_t groups,
                                bool bias)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_(kernel),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      groups_(groups),
      use_bias_(bias) {
    if (groups_ <= 0) {
        throw std::invalid_argument("Groups must be positive");
    }
    if (in_channels_ <= 0 || out_channels_ <= 0) {
        throw std::invalid_argument("Channels must be positive");
    }
    if (kernel_.h <= 0 || kernel_.w <= 0) {
        throw std::invalid_argument("Kernel dimensions must be positive");
    }
    if (stride_.h <= 0 || stride_.w <= 0) {
        throw std::invalid_argument("Stride dimensions must be positive");
    }
    if (dilation_.h <= 0 || dilation_.w <= 0) {
        throw std::invalid_argument("Dilation dimensions must be positive");
    }
    if (padding_.h < 0 || padding_.w < 0) {
        throw std::invalid_argument("Padding dimensions must be non-negative");
    }
    if (in_channels_ % groups_ != 0 || out_channels_ % groups_ != 0) {
        throw std::invalid_argument(
            "In and out channels must be divisible by groups");
    }

    weight_ =
        tensor_t({out_channels_, in_channels_ / groups_, kernel_.h, kernel_.w});
    bias_ = tensor_t({out_channels_}, scalar_t{});

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    for (auto& el : weight_) {
        el = static_cast<scalar_t>(dist(gen));
    }
    if (use_bias_) {
        for (auto& el : bias_) {
            el = static_cast<scalar_t>(dist(gen));
        }
    }
}

template <DType _DType, DeviceLikeType _Device>
Conv2d<_DType, _Device>::Conv2d(std::int64_t in_channels,
                                std::int64_t out_channels, std::int64_t kernel,
                                std::int64_t stride, std::int64_t padding,
                                std::int64_t dilation, std::int64_t groups,
                                bool bias)
    : Conv2d(in_channels, out_channels, {kernel, kernel}, {stride, stride},
             {padding, padding}, {dilation, dilation}, groups, bias) {
}

template <DType _DType, DeviceLikeType _Device>
std::string Conv2d<_DType, _Device>::name() const {
    return "nn.Conv2d";
}

template <DType _DType, DeviceLikeType _Device>
bool Conv2d<_DType, _Device>::load_state(const layer_state_t& state) {
    if (!state.contains("weight")) {
        return false;
    }

    tensor_t weight = state.find("weight");
    tensor_t bias({out_channels_}, scalar_t{});
    if (use_bias_) {
        if (!state.contains("bias")) {
            throw std::invalid_argument("Conv2d layer expects bias tensor");
        }
        bias = std::move(state.find("bias"));
    }

    if (weight.shape() !=
        Shape{out_channels_, in_channels_ / groups_, kernel_.h, kernel_.w}) {
        throw std::invalid_argument("Conv2d weight shape mismatch");
    }
    if (bias.shape() != Shape{out_channels_}) {
        throw std::invalid_argument("Conv2d bias shape mismatch");
    }
    weight_ = std::move(weight);
    bias_ = std::move(bias);

    return true;
}

template <DType _DType, DeviceLikeType _Device>
typename Conv2d<_DType, _Device>::tensor_t Conv2d<_DType, _Device>::forward(
    const tensor_t& input) const {
    if (weight_.shape() !=
        Shape{out_channels_, in_channels_ / groups_, kernel_.h, kernel_.w}) {
        throw std::invalid_argument("Conv2d weight shape mismatch");
    }
    if (use_bias_ && bias_.shape() != Shape{out_channels_}) {
        throw std::invalid_argument("Conv2d bias shape mismatch");
    }

    const auto input_view = input.view();
    const auto weight_view = weight_.view();
    if (!use_bias_) {
        return nn::conv2d<_DType, _Device>(
            input_view, weight_view, stride_.h, stride_.w, padding_.h,
            padding_.w, dilation_.h, dilation_.w, groups_);
    }

    const auto bias_view = bias_.view();
    return nn::conv2d<_DType, _Device>(
        input_view, weight_view, stride_.h, stride_.w, padding_.h, padding_.w,
        dilation_.h, dilation_.w, groups_, std::cref(bias_view));
}

template <DType _DType, DeviceLikeType _Device>
const typename Conv2d<_DType, _Device>::tensor_t&
Conv2d<_DType, _Device>::weight() const {
    return weight_;
}

template <DType _DType, DeviceLikeType _Device>
const typename Conv2d<_DType, _Device>::tensor_t&
Conv2d<_DType, _Device>::bias() const {
    return bias_;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t Conv2d<_DType, _Device>::output_height(
    std::int64_t input_h) const {
    return (input_h + 2 * padding_.h - dilation_.h * (kernel_.h - 1) - 1) /
               stride_.h +
           1;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t Conv2d<_DType, _Device>::output_width(std::int64_t input_w) const {
    return (input_w + 2 * padding_.w - dilation_.w * (kernel_.w - 1) - 1) /
               stride_.w +
           1;
}

}  // namespace nn
}  // namespace fastinf
