#include "fastinf/core/backend/common.hpp"
#include "fastinf/core/tensor.hpp"
#include "fastinf/nn/backend/neon/conv2d.hpp"

namespace fastinf {
namespace nn {
namespace {
template <DType _DType>
Tensor<_DType, DeviceLikeType::neon> conv2d_im2col(
    const TensorView<_DType, DeviceLikeType::neon>& input,
    const TensorView<_DType, DeviceLikeType::neon>& weight, size_2_t stride,
    size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<_DType, DeviceLikeType::neon>>>
        bias) {
    using tensor_t = Tensor<_DType, DeviceLikeType::neon>;
    using scalar_t = typename tensor_t::scalar_t;

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
        throw std::invalid_argument("Out channels must be divisible by groups");
    }
    if (bias.has_value()) {
        const auto& bias_view = bias->get();
        if (bias_view.dim() != 1 || bias_view.shape()[0] != out_channels) {
            throw std::invalid_argument("Conv2d bias shape mismatch");
        }
    }

    const std::int64_t out_h =
        (in_h + 2 * padding.h - dilation.h * (kernel_h - 1) - 1) / stride.h +
        1;
    const std::int64_t out_w =
        (in_w + 2 * padding.w - dilation.w * (kernel_w - 1) - 1) / stride.w +
        1;
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Calculated output size is invalid");
    }

    tensor_t input_tmp, weight_tmp, bias_tmp;
    const scalar_t* input_data =
        backend::ensure_contiguous_data(input, input_tmp);
    const scalar_t* weight_data =
        backend::ensure_contiguous_data(weight, weight_tmp);
    const scalar_t* bias_data = nullptr;
    if (bias.has_value()) {
        if (bias->get().is_contiguous()) {
            bias_data = bias->get().data();
        } else {
            bias_tmp = tensor_t(bias->get());
            bias_data = bias_tmp.data();
        }
    }

    tensor_t output({batch_size, out_channels, out_h, out_w}, scalar_t{});

    const std::int64_t out_hw = out_h * out_w;
    const std::int64_t kernel_size =
        kernel_h * kernel_w * channels_per_group;
    const std::int64_t out_channels_per_group = out_channels / groups;

    for (std::int64_t n = 0; n < batch_size; ++n) {
        for (std::int64_t group = 0; group < groups; ++group) {
            tensor_t im2col_buffer({out_hw, kernel_size}, scalar_t{});
            tensor_t transformed_filter({kernel_size, out_channels_per_group},
                                        scalar_t{});

            const std::int64_t input_channel_offset =
                group * channels_per_group;
            const std::int64_t output_channel_offset =
                group * out_channels_per_group;

            scalar_t* filter_data = transformed_filter.data();
            for (std::int64_t ic = 0; ic < channels_per_group; ++ic) {
                for (std::int64_t kh = 0; kh < kernel_h; ++kh) {
                    for (std::int64_t kw = 0; kw < kernel_w; ++kw) {
                        const std::int64_t filter_row =
                            (ic * kernel_h + kh) * kernel_w + kw;
                        for (std::int64_t oc = 0; oc < out_channels_per_group;
                             ++oc) {
                            const std::int64_t global_oc =
                                output_channel_offset + oc;
                            filter_data[filter_row * out_channels_per_group +
                                        oc] =
                                weight_data[((global_oc * channels_per_group +
                                              ic) *
                                                 kernel_h +
                                             kh) *
                                                kernel_w +
                                            kw];
                        }
                    }
                }
            }

            scalar_t* im2col_data = im2col_buffer.data();
            for (std::int64_t oh = 0; oh < out_h; ++oh) {
                const std::int64_t h_base = oh * stride.h - padding.h;
                for (std::int64_t ow = 0; ow < out_w; ++ow) {
                    const std::int64_t row = oh * out_w + ow;
                    const std::int64_t w_base = ow * stride.w - padding.w;
                    scalar_t* im2col_row = im2col_data + row * kernel_size;

                    for (std::int64_t ic = 0; ic < channels_per_group; ++ic) {
                        const std::int64_t global_ic =
                            input_channel_offset + ic;
                        for (std::int64_t kh = 0; kh < kernel_h; ++kh) {
                            const std::int64_t ih =
                                h_base + kh * dilation.h;
                            for (std::int64_t kw = 0; kw < kernel_w; ++kw) {
                                const std::int64_t col =
                                    (ic * kernel_h + kh) * kernel_w + kw;
                                const std::int64_t iw =
                                    w_base + kw * dilation.w;

                                if (ih >= 0 && ih < in_h && iw >= 0 &&
                                    iw < in_w) {
                                    im2col_row[col] =
                                        input_data[((n * channels + global_ic) *
                                                        in_h +
                                                    ih) *
                                                       in_w +
                                                   iw];
                                }
                            }
                        }
                    }
                }
            }

            tensor_t result = im2col_buffer.mul(transformed_filter);
            const scalar_t* result_data = result.data();
            scalar_t* output_data = output.data();
            for (std::int64_t row = 0; row < out_hw; ++row) {
                const std::int64_t oh = row / out_w;
                const std::int64_t ow = row - oh * out_w;
                for (std::int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    const std::int64_t global_oc = output_channel_offset + oc;
                    scalar_t value =
                        result_data[row * out_channels_per_group + oc];
                    if (bias_data != nullptr) {
                        value += bias_data[global_oc];
                    }
                    output_data[((n * out_channels + global_oc) * out_h + oh) *
                                    out_w +
                                ow] = value;
                }
            }
        }
    }

    return output;
}
}  // namespace

template <>
inline Tensor<DType::int8, DeviceLikeType::neon> conv2d(
    const TensorView<DType::int8, DeviceLikeType::neon>& input,
    const TensorView<DType::int8, DeviceLikeType::neon>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int8, DeviceLikeType::neon>>>
        bias) {
    return conv2d_im2col<DType::int8>(input, weight, stride, padding, dilation,
                                      groups, bias);
}

template <>
inline Tensor<DType::int16, DeviceLikeType::neon> conv2d(
    const TensorView<DType::int16, DeviceLikeType::neon>& input,
    const TensorView<DType::int16, DeviceLikeType::neon>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int16, DeviceLikeType::neon>>>
        bias) {
    return conv2d_im2col<DType::int16>(input, weight, stride, padding,
                                       dilation, groups, bias);
}

template <>
inline Tensor<DType::int32, DeviceLikeType::neon> conv2d(
    const TensorView<DType::int32, DeviceLikeType::neon>& input,
    const TensorView<DType::int32, DeviceLikeType::neon>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int32, DeviceLikeType::neon>>>
        bias) {
    return conv2d_im2col<DType::int32>(input, weight, stride, padding,
                                       dilation, groups, bias);
}

template <>
inline Tensor<DType::int64, DeviceLikeType::neon> conv2d(
    const TensorView<DType::int64, DeviceLikeType::neon>& input,
    const TensorView<DType::int64, DeviceLikeType::neon>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int64, DeviceLikeType::neon>>>
        bias) {
    return conv2d_im2col<DType::int64>(input, weight, stride, padding,
                                       dilation, groups, bias);
}

template <>
inline Tensor<DType::float32, DeviceLikeType::neon> conv2d(
    const TensorView<DType::float32, DeviceLikeType::neon>& input,
    const TensorView<DType::float32, DeviceLikeType::neon>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::float32, DeviceLikeType::neon>>>
        bias) {
    return conv2d_im2col<DType::float32>(input, weight, stride, padding,
                                         dilation, groups, bias);
}

template <>
inline Tensor<DType::float64, DeviceLikeType::neon> conv2d(
    const TensorView<DType::float64, DeviceLikeType::neon>& input,
    const TensorView<DType::float64, DeviceLikeType::neon>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::float64, DeviceLikeType::neon>>>
        bias) {
    return conv2d_im2col<DType::float64>(input, weight, stride, padding,
                                         dilation, groups, bias);
}

template <DType _DType, DeviceLikeType _Device>
ConvIm2Col<_DType, _Device>::ConvIm2Col(std::int64_t in_channels,
                                        std::int64_t out_channels,
                                        size_2_t kernel, size_2_t stride,
                                        size_2_t padding, size_2_t dilation,
                                        std::int64_t groups, bool bias)
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
                                        std::int64_t out_channels,
                                        std::int64_t kernel,
                                        std::int64_t stride,
                                        std::int64_t padding,
                                        std::int64_t dilation,
                                        std::int64_t groups, bool bias)
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
typename ConvIm2Col<_DType, _Device>::tensor_t
ConvIm2Col<_DType, _Device>::forward(const tensor_t& input) const {
    if (weight_.shape() !=
        Shape{out_channels_, in_channels_ / groups_, kernel_.h, kernel_.w}) {
        throw std::invalid_argument("ConvIm2Col weight shape mismatch");
    }
    if (use_bias_ && bias_.shape() != Shape{out_channels_}) {
        throw std::invalid_argument("ConvIm2Col bias shape mismatch");
    }

    const auto input_view = input.view();
    const auto weight_view = weight_.view();
    if (!use_bias_) {
        return nn::conv2d<_DType, _Device>(input_view, weight_view, stride_,
                                           padding_, dilation_, groups_);
    }

    const auto bias_view = bias_.view();
    return nn::conv2d<_DType, _Device>(input_view, weight_view, stride_,
                                       padding_, dilation_, groups_,
                                       std::cref(bias_view));
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
std::int64_t ConvIm2Col<_DType, _Device>::output_width(
    std::int64_t input_w) const {
    return (input_w + 2 * padding_.w - dilation_.w * (kernel_.w - 1) - 1) /
               stride_.w +
           1;
}

}  // namespace nn
}  // namespace fastinf
