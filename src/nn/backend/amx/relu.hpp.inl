#include <Accelerate/Accelerate.h>

#include "fastinf/core/tensor.hpp"
#include "fastinf/nn/backend/amx/relu.hpp"

namespace fastinf {
namespace nn {
template <>
inline Tensor<DType::float32, DeviceLikeType::amx> relu(
    const TensorView<DType::float32, DeviceLikeType::amx>& input) {
    using tensor_t = Tensor<DType::float32, DeviceLikeType::amx>;
    using scalar_t = typename tensor_t::scalar_t;

    tensor_t output(input);
    scalar_t* output_data = output.data();
    const scalar_t zero = scalar_t{};
    vDSP_vthres(output_data, 1, &zero, output_data, 1,
                static_cast<vDSP_Length>(input.numel()));

    return output;
}

template <>
inline Tensor<DType::float64, DeviceLikeType::amx> relu(
    const TensorView<DType::float64, DeviceLikeType::amx>& input) {
    using tensor_t = Tensor<DType::float64, DeviceLikeType::amx>;
    using scalar_t = typename tensor_t::scalar_t;

    tensor_t output(input);
    scalar_t* output_data = output.data();
    const scalar_t zero = scalar_t{};
    vDSP_vthresD(output_data, 1, &zero, output_data, 1,
                 static_cast<vDSP_Length>(input.numel()));

    return output;
}

}  // namespace nn
}  // namespace fastinf
