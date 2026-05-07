#include <Accelerate/Accelerate.h>

#include "fastinf/core/tensor.hpp"
#include "fastinf/nn/backend/amx/conv2d.hpp"

namespace fastinf {
namespace nn {
namespace {
template <DType _DType>
struct BNNSTraits;

template <>
struct BNNSTraits<DType::float32> {
    static constexpr BNNSDataType value = BNNSDataTypeFloat32;
};

template <>
struct BNNSTraits<DType::int8> {
    static constexpr BNNSDataType value = BNNSDataTypeInt8;
};

template <>
struct BNNSTraits<DType::int16> {
    static constexpr BNNSDataType value = BNNSDataTypeInt16;
};

template <>
struct BNNSTraits<DType::int32> {
    static constexpr BNNSDataType value = BNNSDataTypeInt32;
};

template <>
struct BNNSTraits<DType::int64> {
    static constexpr BNNSDataType value = BNNSDataTypeInt64;
};

template <DType _DType>
Tensor<_DType, DeviceLikeType::amx> conv2d_amx(
    const TensorView<_DType, DeviceLikeType::amx>& input,
    const TensorView<_DType, DeviceLikeType::amx>& weight, size_2_t stride,
    size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<
        std::reference_wrapper<const TensorView<_DType, DeviceLikeType::amx>>>
        bias) {
    using tensor_t = Tensor<_DType, DeviceLikeType::amx>;
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
        (in_h + 2 * padding.h - dilation.h * (kernel_h - 1) - 1) / stride.h + 1;
    const std::int64_t out_w =
        (in_w + 2 * padding.w - dilation.w * (kernel_w - 1) - 1) / stride.w + 1;
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Calculated output size is invalid");
    }

    tensor_t input_tmp, weight_tmp, bias_tmp;
    scalar_t* input_data = backend::ensure_contiguous_data(input, input_tmp);
    scalar_t* weight_data = backend::ensure_contiguous_data(weight, weight_tmp);
    scalar_t* bias_data = nullptr;

    tensor_t output({batch_size, out_channels, out_h, out_w}, scalar_t{});

    BNNSNDArrayDescriptor input_bnns = {};
    input_bnns.layout = BNNSDataLayoutImageCHW;
    input_bnns.size[0] = static_cast<std::size_t>(in_w);
    input_bnns.size[1] = static_cast<std::size_t>(in_h);
    input_bnns.size[2] = static_cast<std::size_t>(channels);
    input_bnns.size[3] = static_cast<std::size_t>(batch_size);
    input_bnns.data_type = BNNSTraits<_DType>::value;

    BNNSNDArrayDescriptor output_bnns = {};
    output_bnns.layout = BNNSDataLayoutImageCHW;
    output_bnns.size[0] = static_cast<std::size_t>(out_w);
    output_bnns.size[1] = static_cast<std::size_t>(out_h);
    output_bnns.size[2] = static_cast<std::size_t>(out_channels);
    output_bnns.size[3] = static_cast<std::size_t>(batch_size);
    output_bnns.data_type = BNNSTraits<_DType>::value;

    BNNSNDArrayDescriptor weight_bnns = {};
    weight_bnns.layout = BNNSDataLayoutConvolutionWeightsOIHW;
    weight_bnns.size[0] = static_cast<std::size_t>(kernel_w);
    weight_bnns.size[1] = static_cast<std::size_t>(kernel_h);
    weight_bnns.size[2] = static_cast<std::size_t>(channels_per_group);
    weight_bnns.size[3] = static_cast<std::size_t>(out_channels);
    weight_bnns.data = weight_data;
    weight_bnns.data_type = BNNSTraits<_DType>::value;

    BNNSLayerParametersConvolution params_bnns = {};
    params_bnns.i_desc = input_bnns;
    params_bnns.o_desc = output_bnns;
    params_bnns.w_desc = weight_bnns;

    if (bias.has_value()) {
        if (bias->get().is_contiguous()) {
            bias_data = bias->get().data();
        } else {
            bias_tmp = tensor_t(bias->get());
            bias_data = bias_tmp.data();
        }

        BNNSNDArrayDescriptor bias_bnns = {};
        bias_bnns.layout = BNNSDataLayoutVector;
        bias_bnns.size[0] = static_cast<std::size_t>(out_channels);
        bias_bnns.data = bias_data;
        bias_bnns.data_type = BNNSTraits<_DType>::value;

        params_bnns.bias = bias_bnns;
    }

    params_bnns.x_stride = static_cast<std::size_t>(stride.w);
    params_bnns.y_stride = static_cast<std::size_t>(stride.h);
    params_bnns.x_dilation_stride = static_cast<std::size_t>(dilation.w);
    params_bnns.y_dilation_stride = static_cast<std::size_t>(dilation.h);
    params_bnns.x_padding = static_cast<std::size_t>(padding.w);
    params_bnns.y_padding = static_cast<std::size_t>(padding.h);
    params_bnns.groups = static_cast<std::size_t>(groups);

    BNNSFilter filter = BNNSFilterCreateLayerConvolution(&params_bnns, NULL);
    if (filter == nullptr) {
        throw std::runtime_error("BNNSFilterCreateLayerConvolution failed");
    }

    const int status = BNNSFilterApply(filter, input_data, output.data());
    BNNSFilterDestroy(filter);
    if (status != 0) {
        throw std::runtime_error("BNNSFilterApply failed");
    }

    return output;
}
}  // namespace

template <>
inline Tensor<DType::int8, DeviceLikeType::amx> conv2d(
    const TensorView<DType::int8, DeviceLikeType::amx>& input,
    const TensorView<DType::int8, DeviceLikeType::amx>& weight, size_2_t stride,
    size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int8, DeviceLikeType::amx>>>
        bias) {
    return conv2d_amx<DType::int8>(input, weight, stride, padding, dilation,
                                   groups, bias);
}

template <>
inline Tensor<DType::int16, DeviceLikeType::amx> conv2d(
    const TensorView<DType::int16, DeviceLikeType::amx>& input,
    const TensorView<DType::int16, DeviceLikeType::amx>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int16, DeviceLikeType::amx>>>
        bias) {
    return conv2d_amx<DType::int16>(input, weight, stride, padding, dilation,
                                    groups, bias);
}

template <>
inline Tensor<DType::int32, DeviceLikeType::amx> conv2d(
    const TensorView<DType::int32, DeviceLikeType::amx>& input,
    const TensorView<DType::int32, DeviceLikeType::amx>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int32, DeviceLikeType::amx>>>
        bias) {
    return conv2d_amx<DType::int32>(input, weight, stride, padding, dilation,
                                    groups, bias);
}

template <>
inline Tensor<DType::int64, DeviceLikeType::amx> conv2d(
    const TensorView<DType::int64, DeviceLikeType::amx>& input,
    const TensorView<DType::int64, DeviceLikeType::amx>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::int64, DeviceLikeType::amx>>>
        bias) {
    return conv2d_amx<DType::int64>(input, weight, stride, padding, dilation,
                                    groups, bias);
}

template <>
inline Tensor<DType::float32, DeviceLikeType::amx> conv2d(
    const TensorView<DType::float32, DeviceLikeType::amx>& input,
    const TensorView<DType::float32, DeviceLikeType::amx>& weight,
    size_2_t stride, size_2_t padding, size_2_t dilation, std::int64_t groups,
    std::optional<std::reference_wrapper<
        const TensorView<DType::float32, DeviceLikeType::amx>>>
        bias) {
    return conv2d_amx<DType::float32>(input, weight, stride, padding, dilation,
                                      groups, bias);
}

}  // namespace nn
}  // namespace fastinf
