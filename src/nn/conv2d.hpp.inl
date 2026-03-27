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
typename Conv2d<_DType, _Device>::tensor_t Conv2d<_DType, _Device>::forward(
    const tensor_t& input) const {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [N, C, H, W]");
    }

    const auto& in_shape = input.shape();
    const auto batch_size = in_shape[0];
    const auto channels = in_shape[1];
    const auto in_h = in_shape[2];
    const auto in_w = in_shape[3];

    if (channels != in_channels_) {
        throw std::invalid_argument("Input channels do not match Conv2d");
    }

    const std::int64_t out_h = output_height(in_h);
    const std::int64_t out_w = output_width(in_w);
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Calculated output size is invalid");
    }

    tensor_t output({batch_size, out_channels_, out_h, out_w}, scalar_t{});

    const std::int64_t channels_per_group = in_channels_ / groups_;
    const std::int64_t out_channels_per_group = out_channels_ / groups_;

    for (std::int64_t n = 0; n < batch_size; ++n) {
        for (std::int64_t oc = 0; oc < out_channels_; ++oc) {
            const std::int64_t group_idx = oc / out_channels_per_group;
            const std::int64_t in_group_offset = group_idx * channels_per_group;

            for (std::int64_t oh = 0; oh < out_h; ++oh) {
                for (std::int64_t ow = 0; ow < out_w; ++ow) {
                    scalar_t acc{};
                    const auto h_start = (oh * stride_.h) - (padding_.h);
                    const auto w_start = (ow * stride_.w) - (padding_.w);

                    for (std::int64_t ic = 0; ic < channels_per_group; ++ic) {
                        for (std::int64_t kh = 0; kh < kernel_.h; ++kh) {
                            const auto ih = h_start + kh * dilation_.h;
                            if (ih < 0 || ih >= in_h) {
                                continue;
                            }
                            for (std::int64_t kw = 0; kw < kernel_.w; ++kw) {
                                const auto iw = w_start + kw * dilation_.w;
                                if (iw < 0 || iw >= in_w) {
                                    continue;
                                }

                                acc += input.at(
                                           {n, in_group_offset + ic, ih, iw}) *
                                       weight_.at({oc, ic, kh, kw});
                            }
                        }
                    }

                    if (use_bias_) {
                        acc += bias_.at({oc});
                    }

                    output.at({n, oc, oh, ow}) = acc;
                }
            }
        }
    }

    return output;
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
void Conv2d<_DType, _Device>::load_weights(const tensor_t& weight,
                                           const tensor_t& bias) {
    if (weight.shape() !=
        Shape{out_channels_, in_channels_ / groups_, kernel_.h, kernel_.w}) {
        throw std::invalid_argument("Conv2d weight shape mismatch");
    }
    if (bias.shape() != Shape{out_channels_}) {
        throw std::invalid_argument("Conv2d bias shape mismatch");
    }
    weight_ = weight;
    bias_ = bias;
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
