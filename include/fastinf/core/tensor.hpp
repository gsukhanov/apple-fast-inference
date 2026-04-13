#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <opencv2/core/mat.hpp>

#include "device.hpp"
#include "dtype.hpp"
#include "ops.hpp"
#include "tensor_iterator.hpp"
#include "tensor_printer.hpp"
#include "tensor_view.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class Tensor {
 public:
    using scalar_t = typename DTypeTraits<_DType>::type;
    using view_t = TensorView<_DType, _Device>;

    using Iterator = TensorIterator<scalar_t>;
    using ConstIterator = TensorIterator<const scalar_t>;
    Iterator begin();
    Iterator end();
    ConstIterator begin() const;
    ConstIterator end() const;
    ConstIterator cbegin() const;
    ConstIterator cend() const;

    Tensor() = default;
    explicit Tensor(Shape shape, const scalar_t& value = scalar_t{});
    Tensor(std::initializer_list<std::int64_t> shape,
           const scalar_t value = scalar_t{});
    Tensor(Shape shape, std::vector<scalar_t> data);
    explicit Tensor(const view_t& view);
    Tensor(const cv::Mat& m);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    view_t view() const;

    const Shape& shape() const;

    scalar_t* data();
    const scalar_t* data() const;

    const TensorDesc& desc() const;

    bool is_contiguous() const;
    int dim() const;

    scalar_t& at(const Shape& indices);
    const scalar_t& at(const Shape& indices) const;
    view_t slice(const Shape& indices) const;

    Tensor contiguous() const;
    Tensor clone() const;
    view_t transpose(std::size_t dim0, std::size_t dim1) const;
    view_t t() const;

    Tensor& operator+=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(const view_t& other);
    Tensor operator+(const view_t& other) const;
    Tensor& operator+=(scalar_t scalar);
    Tensor operator+(scalar_t scalar) const;

    Tensor& operator-=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;
    Tensor& operator-=(const view_t& other);
    Tensor operator-(const view_t& other) const;
    Tensor& operator-=(scalar_t scalar);
    Tensor operator-(scalar_t scalar) const;

    Tensor& operator*=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor& operator*=(const view_t& other);
    Tensor operator*(const view_t& other) const;
    Tensor& operator*=(scalar_t scalar);
    Tensor operator*(scalar_t scalar) const;

    Tensor mul(const Tensor& other) const;
    Tensor mul(const view_t& other) const;

 private:
    scalar_t* data_{nullptr};
    TensorDesc desc_;
};

template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
std::ostream& operator<<(std::ostream& os,
                         const Tensor<_DType, _Device>& tensor);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator+(const TensorView<_DType, _Device>& lhs,
                                  const TensorView<_DType, _Device>& rhs);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator+(
    const TensorView<_DType, _Device>& input,
    typename TensorView<_DType, _Device>::scalar_t scalar);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator+(
    typename TensorView<_DType, _Device>::scalar_t scalar,
    const TensorView<_DType, _Device>& input);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator-(const TensorView<_DType, _Device>& lhs,
                                  const TensorView<_DType, _Device>& rhs);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator-(
    const TensorView<_DType, _Device>& input,
    typename TensorView<_DType, _Device>::scalar_t scalar);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator*(const TensorView<_DType, _Device>& lhs,
                                  const TensorView<_DType, _Device>& rhs);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator*(
    const TensorView<_DType, _Device>& input,
    typename TensorView<_DType, _Device>::scalar_t scalar);

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator*(
    typename TensorView<_DType, _Device>::scalar_t scalar,
    const TensorView<_DType, _Device>& input);

};  // namespace fastinf

#include "../../../src/core/tensor.hpp.inl"
#include "backend/amx/matmul.hpp"
