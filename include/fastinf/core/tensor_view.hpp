#pragma once

#include <cstdint>

#include "device.hpp"
#include "dtype.hpp"
#include "tensor_desc.hpp"
#include "tensor_iterator.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
class Tensor;

template <DType _DType, DeviceLikeType _Device>
class TensorView {
 public:
    using scalar_t = typename DTypeTraits<_DType>::type;
    using Iterator = TensorIterator<scalar_t>;
    using ConstIterator = TensorIterator<const scalar_t>;

    TensorView() = default;

    TensorView(scalar_t* data, TensorDesc desc);
    explicit TensorView(const Tensor<_DType, _Device>& tensor);

    Iterator begin();
    Iterator end();
    ConstIterator begin() const;
    ConstIterator end() const;
    ConstIterator cbegin() const;
    ConstIterator cend() const;

    scalar_t* data() const;
    const TensorDesc& desc() const;

    const Shape& shape() const;
    const Strides& strides() const;

    std::int64_t numel() const;
    int dim() const;

    scalar_t& at(const Shape& indices);
    const scalar_t& at(const Shape& indices) const;

    bool is_contiguous() const;

    TensorView& operator=(const TensorView& other);
    TensorView& operator=(const Tensor<_DType, _Device>& other);

    TensorView slice(const Shape& indices) const;
    TensorView permute(const Shape& perm) const;
    TensorView transpose(std::size_t dim0, std::size_t dim1) const;
    TensorView t() const;

    TensorView& operator+=(const TensorView& other);
    TensorView& operator+=(scalar_t scalar);

    TensorView& operator-=(const TensorView& other);
    TensorView& operator-=(scalar_t scalar);

    TensorView& operator*=(const TensorView& other);
    TensorView& operator*=(scalar_t scalar);

    Tensor<_DType, _Device> mul(const TensorView& other) const;

 private:
    scalar_t* data_{nullptr};
    TensorDesc desc_;

    std::size_t offset_for(const Shape& indices) const;
};

};  // namespace fastinf

#include "../../../src/core/tensor_view.hpp.inl"
