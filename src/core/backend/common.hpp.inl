#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace backend {
template <DType _DType, DeviceLikeType _Device>
typename DTypeTraits<_DType>::type* ensure_contiguous_data(
    const TensorView<_DType, _Device>& view, Tensor<_DType, _Device>& tmp) {
    if (view.is_contiguous()) {
        return view.data();
    }

    tmp = Tensor<_DType, _Device>(view);
    return tmp.data();
}
}  // namespace backend
}  // namespace fastinf
