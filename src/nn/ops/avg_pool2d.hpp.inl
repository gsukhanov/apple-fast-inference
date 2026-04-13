#include "fastinf/nn/ops/avg_pool2d.hpp"

#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> avg_pool2d(const TensorView<_DType, _Device>& input,
                                   std::int64_t kernel_h,
                                   std::int64_t kernel_w,
                                   std::int64_t stride_h,
                                   std::int64_t stride_w,
                                   std::int64_t padding_h,
                                   std::int64_t padding_w) {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D [N, C, H, W]");
    }

    const auto& in_shape = input.shape();
    const auto batch_size = in_shape[0];
    const auto channels = in_shape[1];
    const auto in_h = in_shape[2];
    const auto in_w = in_shape[3];

    const std::int64_t out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    const std::int64_t out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Calculated output size is invalid");
    }

    Tensor<_DType, _Device> output({batch_size, channels, out_h, out_w},
                                   typename Tensor<_DType, _Device>::scalar_t{});

    for (std::int64_t n = 0; n < batch_size; ++n) {
        for (std::int64_t c = 0; c < channels; ++c) {
            for (std::int64_t oh = 0; oh < out_h; ++oh) {
                for (std::int64_t ow = 0; ow < out_w; ++ow) {
                    const auto h_start = oh * stride_h - padding_h;
                    const auto w_start = ow * stride_w - padding_w;

                    typename Tensor<_DType, _Device>::scalar_t sum = 0;
                    std::int64_t valid_count = 0;

                    for (std::int64_t kh = 0; kh < kernel_h; ++kh) {
                        const auto ih = h_start + kh;
                        if (ih < 0 || ih >= in_h) {
                            continue;
                        }
                        for (std::int64_t kw = 0; kw < kernel_w; ++kw) {
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

                    output.at({n, c, oh, ow}) =
                        static_cast<typename Tensor<_DType, _Device>::scalar_t>(
                            static_cast<double>(sum) / valid_count);
                }
            }
        }
    }

    return output;
}
}  // namespace nn
}  // namespace fastinf
