#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "device.hpp"
#include "dtype.hpp"
#include "tensor_view.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class Tensor {
 public:
    using scalar_t = typename DTypeTraits<_DType>::type;
    using view_t = TensorView<_DType, _Device>;

    Tensor() = default;
    explicit Tensor(Shape shape, const scalar_t& value = scalar_t{});
    Tensor(std::initializer_list<std::int64_t> shape,
           const scalar_t value = scalar_t{});
    Tensor(Shape shape, std::vector<scalar_t> data);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    view_t view() const;

    scalar_t* data();
    const scalar_t* data() const;

    const TensorDesc& desc() const;

    bool is_contiguous() const;
    int dim() const;

    scalar_t& at(const Shape& indices);
    const scalar_t& at(const Shape& indices) const;

    Tensor clone() const;

    Tensor& operator+=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(scalar_t scalar);
    Tensor operator+(scalar_t scalar) const;

    Tensor& operator-=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;
    Tensor& operator-=(scalar_t scalar);
    Tensor operator-(scalar_t scalar) const;

    Tensor& operator*=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor& operator*=(scalar_t scalar);
    Tensor operator*(scalar_t scalar) const;

    Tensor mul(const Tensor& other) const;

 private:
    scalar_t* data_{nullptr};
    TensorDesc desc_;

    std::size_t offset_for(const Shape& indices) const;
};

template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
std::ostream& operator<<(std::ostream& os,
                         const Tensor<_DType, _Device>& tensor);

};  // namespace fastinf

#include "../../../src/core/tensor.hpp.inl"
