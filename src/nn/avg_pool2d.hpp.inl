#include "fastinf/nn/avg_pool2d.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
AvgPool2d<_DType, _Device>::AvgPool2d(size_2_t kernel, size_2_t stride,
                                      size_2_t padding)
    : kernel_(kernel), stride_(stride), padding_(padding) {
    if (kernel_.h <= 0 || kernel_.w <= 0) {
        throw std::invalid_argument("Kernel dimensions must be positive");
    }
    if (stride_.h <= 0 || stride_.w <= 0) {
        throw std::invalid_argument("Stride dimensions must be positive");
    }
    if (padding_.h < 0 || padding_.w < 0) {
        throw std::invalid_argument("Padding dimensions must be non-negative");
    }
}

template <DType _DType, DeviceLikeType _Device>
AvgPool2d<_DType, _Device>::AvgPool2d(std::int64_t kernel, std::int64_t stride,
                                      std::int64_t padding)
    : AvgPool2d({kernel, kernel}, {stride, stride}, {padding, padding}) {
}

template <DType _DType, DeviceLikeType _Device>
std::string AvgPool2d<_DType, _Device>::name() const {
    return "nn.AvgPool2d";
}

template <DType _DType, DeviceLikeType _Device>
typename AvgPool2d<_DType, _Device>::tensor_t
AvgPool2d<_DType, _Device>::forward(const tensor_t& input) const {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [N, C, H, W]");
    }

    const auto& in_shape = input.shape();
    const auto batch_size = in_shape[0];
    const auto channels = in_shape[1];
    const auto in_h = in_shape[2];
    const auto in_w = in_shape[3];

    const std::int64_t out_h = output_height(in_h);
    const std::int64_t out_w = output_width(in_w);
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Calculated output size is invalid");
    }

    tensor_t output({batch_size, channels, out_h, out_w}, scalar_t{});

    for (std::int64_t n = 0; n < batch_size; ++n) {
        for (std::int64_t c = 0; c < channels; ++c) {
            for (std::int64_t oh = 0; oh < out_h; ++oh) {
                for (std::int64_t ow = 0; ow < out_w; ++ow) {
                    const auto h_start = oh * stride_.h - padding_.h;
                    const auto w_start = ow * stride_.w - padding_.w;

                    scalar_t sum = 0;
                    std::int64_t valid_count = 0;

                    for (std::int64_t kh = 0; kh < kernel_.h; ++kh) {
                        const auto ih = h_start + kh;
                        if (ih < 0 || ih >= in_h) {
                            continue;
                        }
                        for (std::int64_t kw = 0; kw < kernel_.w; ++kw) {
                            const auto iw = w_start + kw;
                            if (iw < 0 || iw >= in_w) {
                                continue;
                            }

                            sum += input.at({n, c, ih, iw});
                            ++valid_count;
                        }
                    }

                    if (valid_count == 0) {
                        continue;
                    }

                    output.at({n, c, oh, ow}) = static_cast<scalar_t>(
                        static_cast<double>(sum) / valid_count);
                }
            }
        }
    }

    return output;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t AvgPool2d<_DType, _Device>::output_height(
    std::int64_t input_h) const {
    const auto num = input_h + 2 * padding_.h - kernel_.h;
    if (num < 0) {
        throw std::invalid_argument(
            "Kernel height larger than padded input height");
    }

    return num / stride_.h + 1;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t AvgPool2d<_DType, _Device>::output_width(
    std::int64_t input_w) const {
    const auto num = input_w + 2 * padding_.w - kernel_.w;
    if (num < 0) {
        throw std::invalid_argument(
            "Kernel width larger than padded input width");
    }

    return num / stride_.w + 1;
}
}  // namespace nn
}  // namespace fastinf
