#include "fastinf/nn/backend/greedy/conv2d.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
ConvIm2Col<_DType, _Device>::ConvIm2Col(std::int64_t in_channels,
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
ConvIm2Col<_DType, _Device>::ConvIm2Col(std::int64_t in_channels,
                                std::int64_t out_channels, std::int64_t kernel,
                                std::int64_t stride, std::int64_t padding,
                                std::int64_t dilation, std::int64_t groups,
                                bool bias)
    : ConvIm2Col(in_channels, out_channels, {kernel, kernel}, {stride, stride},
             {padding, padding}, {dilation, dilation}, groups, bias) {
}

template <DType _DType, DeviceLikeType _Device>
std::string ConvIm2Col<_DType, _Device>::name() const {
    return "nn.ConvIm2Col";
}

template <DType _DType, DeviceLikeType _Device>
bool ConvIm2Col<_DType, _Device>::load_state(const layer_state_t& state) {
    if (!state.contains("weight")) {
        return false;
    }

    tensor_t weight = state.find("weight");
    tensor_t bias({out_channels_}, scalar_t{});
    if (use_bias_) {
        if (!state.contains("bias")) {
            throw std::invalid_argument("ConvIm2Col layer expects bias tensor");
        }
        bias = std::move(state.find("bias"));
    }

    if (weight.shape() !=
        Shape{out_channels_, in_channels_ / groups_, kernel_.h, kernel_.w}) {
        throw std::invalid_argument("ConvIm2Col weight shape mismatch");
    }
    if (bias.shape() != Shape{out_channels_}) {
        throw std::invalid_argument("ConvIm2Col bias shape mismatch");
    }
    weight_ = std::move(weight);
    bias_ = std::move(bias);

    return true;
}

template <DType _DType, DeviceLikeType _Device>
typename ConvIm2Col<_DType, _Device>::tensor_t ConvIm2Col<_DType, _Device>::forward(
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
        throw std::invalid_argument("Input channels do not match ConvIm2Col");
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
        for (std::int64_t gn = 0; gn < groups_; ++gn) {
            // im2col:
            tensor_t im2col_buffer({out_h * out_w, kernel_.h * kernel_.w * channels_per_group});
            tensor_t transformed_filter({kernel_.h * kernel_.w * channels_per_group, out_channels_per_group});

            const std::int64_t in_out_group_offset = gn * out_channels_per_group;
            const std::int64_t in_group_offset = gn * channels_per_group;

            for (std::int64_t oc = 0; oc < out_channels_per_group; ++oc) {            
                for (std::int64_t ic = 0; ic < channels_per_group; ++ic) {
                    std::int64_t current_out_channel = oc + in_out_group_offset;
                    std::int64_t current_channel = ic + in_group_offset;

                    for (std::int64_t oh = 0; oh < out_h; ++oh) {
                        for (std::int64_t ow = 0; ow < out_w; ++ow) {

                            for (std::int64_t kh = 0; kh < kernel_.h; ++kh) {
                                for (std::int64_t kw = 0; kw < kernel_.w; ++kw) {

                                    std::int64_t w_buffer = ow + oh * out_w;
                                    std::int64_t h_buffer_w_filter = current_channel + (kw + kh * kernel_.w) * channels_per_group;

                                    std::int64_t h_input = oh * stride_.h + kh * dilation_.h - padding_.h;
                                    std::int64_t w_input = ow * stride_.w + kw * dilation_.w - padding_.w;
                                    
                                    im2col_buffer.at({w_buffer, h_buffer_w_filter}) =
                                        (h_input < 0 || w_input < 0 || h_input >= in_h || w_input >= in_w) ?
                                        0 : input.at({n, current_channel, h_input, w_input});

                                    transformed_filter.at({h_buffer_w_filter, current_out_channel}) =
                                        weight().at({current_out_channel, ic, kh, kw});

                                }
                            }
                        }
                    }
                }
            }
            // convolution:
            tensor_t result = im2col_buffer.mul(transformed_filter);
            for (std::int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                for (std::int64_t oh = 0; oh < out_h; ++oh) {
                    for (std::int64_t ow = 0; ow < out_w; ++ow) {
                        std::int64_t current_out_channel = oc + in_out_group_offset;
                        std::int64_t w_result = ow + oh * out_w;
                        output.at({n, current_out_channel, oh, ow}) = result.at({w_result, current_out_channel});
                    }
                }
            }
        }
    }

    return output;
}

template <DType _DType, DeviceLikeType _Device>
const typename ConvIm2Col<_DType, _Device>::tensor_t&
ConvIm2Col<_DType, _Device>::weight() const {
    return weight_;
}

template <DType _DType, DeviceLikeType _Device>
const typename ConvIm2Col<_DType, _Device>::tensor_t&
ConvIm2Col<_DType, _Device>::bias() const {
    return bias_;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t ConvIm2Col<_DType, _Device>::output_height(
    std::int64_t input_h) const {
    return (input_h + 2 * padding_.h - dilation_.h * (kernel_.h - 1) - 1) /
               stride_.h +
           1;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t ConvIm2Col<_DType, _Device>::output_width(std::int64_t input_w) const {
    return (input_w + 2 * padding_.w - dilation_.w * (kernel_.w - 1) - 1) /
               stride_.w +
           1;
}

}  // namespace nn
}  // namespace fastinf
