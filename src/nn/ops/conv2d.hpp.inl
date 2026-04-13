#include "fastinf/nn/ops/conv2d.hpp"

#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> conv2d(
    const TensorView<_DType, _Device>& input,
    const TensorView<_DType, _Device>& weight, std::int64_t stride_h,
    std::int64_t stride_w, std::int64_t padding_h, std::int64_t padding_w,
    std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t groups,
    std::optional<std::reference_wrapper<const TensorView<_DType, _Device>>> bias) {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [N, C, H, W]");
    }
    if (weight.dim() != 4) {
        throw std::invalid_argument(
            "Conv2d weight expects 4D shape [O, I/groups, KH, KW]");
    }
    if (groups <= 0) {
        throw std::invalid_argument("Groups must be positive");
    }

    const auto& in_shape = input.shape();
    const auto batch_size = in_shape[0];
    const auto channels = in_shape[1];
    const auto in_h = in_shape[2];
    const auto in_w = in_shape[3];

    const auto& weight_shape = weight.shape();
    const auto out_channels = weight_shape[0];
    const auto channels_per_group = weight_shape[1];
    const auto kernel_h = weight_shape[2];
    const auto kernel_w = weight_shape[3];

    if (channels != channels_per_group * groups) {
        throw std::invalid_argument("Input channels do not match Conv2d");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument(
            "Out channels must be divisible by groups");
    }
    if (bias.has_value()) {
        const auto& bias_view = bias->get();
        if (bias_view.dim() != 1 || bias_view.shape()[0] != out_channels) {
            throw std::invalid_argument("Conv2d bias shape mismatch");
        }
    }

    const std::int64_t out_h =
        (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const std::int64_t out_w =
        (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Calculated output size is invalid");
    }

    Tensor<_DType, _Device> output({batch_size, out_channels, out_h, out_w},
                                   typename Tensor<_DType, _Device>::scalar_t{});

    const std::int64_t out_channels_per_group = out_channels / groups;

    for (std::int64_t n = 0; n < batch_size; ++n) {
        for (std::int64_t oc = 0; oc < out_channels; ++oc) {
            const std::int64_t group_idx = oc / out_channels_per_group;
            const std::int64_t in_group_offset = group_idx * channels_per_group;

            for (std::int64_t oh = 0; oh < out_h; ++oh) {
                for (std::int64_t ow = 0; ow < out_w; ++ow) {
                    typename Tensor<_DType, _Device>::scalar_t acc{};
                    const auto h_start = (oh * stride_h) - padding_h;
                    const auto w_start = (ow * stride_w) - padding_w;

                    for (std::int64_t ic = 0; ic < channels_per_group; ++ic) {
                        for (std::int64_t kh = 0; kh < kernel_h; ++kh) {
                            const auto ih = h_start + kh * dilation_h;
                            if (ih < 0 || ih >= in_h) {
                                continue;
                            }
                            for (std::int64_t kw = 0; kw < kernel_w; ++kw) {
                                const auto iw = w_start + kw * dilation_w;
                                if (iw < 0 || iw >= in_w) {
                                    continue;
                                }

                                acc += input.at({n, in_group_offset + ic, ih, iw}) *
                                       weight.at({oc, ic, kh, kw});
                            }
                        }
                    }

                    if (bias.has_value()) {
                        acc += bias->get().at({oc});
                    }

                    output.at({n, oc, oh, ow}) = acc;
                }
            }
        }
    }

    return output;
}
}  // namespace nn
}  // namespace fastinf
