#pragma once

#include <cstdint>

#include "device.hpp"
#include "dtype.hpp"
#include "tensor_desc.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class TensorView {
 public:
    using scalar_t = typename DTypeTraits<_DType>::type;

    TensorView() = default;

    TensorView(scalar_t* data, TensorDesc desc);

    scalar_t* data() const;
    const TensorDesc& desc() const;

    const Shape& shape() const;
    const Strides& strides() const;

    std::int64_t numel() const;

    bool is_contiguous() const;

    TensorView permute(const Shape& perm);

 private:
    scalar_t* data_{nullptr};
    TensorDesc desc_;
};

};  // namespace fastinf

#include "../../../src/core/tensor_view.hpp.inl"