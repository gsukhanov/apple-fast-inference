#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace {
#if FASTINF_HAS_OPENCV
template <DType _DType>
struct OpenCvTraits;

template <>
struct OpenCvTraits<DType::int8> {
    static constexpr int depth = CV_8S;
};

template <>
struct OpenCvTraits<DType::int16> {
    static constexpr int depth = CV_16S;
};

template <>
struct OpenCvTraits<DType::int32> {
    static constexpr int depth = CV_32S;
};

template <>
struct OpenCvTraits<DType::float32> {
    static constexpr int depth = CV_32F;
};

template <>
struct OpenCvTraits<DType::float64> {
    static constexpr int depth = CV_64F;
};
#endif
}  // namespace

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::Iterator Tensor<_DType, _Device>::begin() {
    return Iterator(data_, &desc_);
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::Iterator Tensor<_DType, _Device>::end() {
    return Iterator(data_, &desc_, true);
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::ConstIterator Tensor<_DType, _Device>::begin()
    const {
    return ConstIterator(data_, &desc_);
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::ConstIterator Tensor<_DType, _Device>::end()
    const {
    return ConstIterator(data_, &desc_, true);
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::ConstIterator
Tensor<_DType, _Device>::cbegin() const {
    return begin();
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::ConstIterator Tensor<_DType, _Device>::cend()
    const {
    return end();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(Shape shape, const scalar_t& value)
    : desc_(shape, make_contiguous_strides(shape)) {
    const auto n = static_cast<std::size_t>(desc_.numel());
    data_ = new scalar_t[n];
    for (std::size_t i = 0; i < n; ++i) {
        data_[i] = value;
    }
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(std::initializer_list<std::int64_t> shape,
                                const scalar_t value)
    : Tensor(Shape(shape), value) {
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(Shape shape, std::vector<scalar_t> data)
    : desc_(shape, make_contiguous_strides(shape)) {
    const auto n = static_cast<std::size_t>(desc_.numel());
    data_ = new scalar_t[n];
    for (std::size_t i = 0; i < n; ++i) {
        data_[i] = data[i];
    }
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(const view_t& view)
    : desc_(view.shape(), make_contiguous_strides(view.shape())) {
    const auto n = static_cast<std::size_t>(desc_.numel());
    data_ = new scalar_t[n];
    auto it = view.begin();
    for (std::size_t i = 0; i < n; ++i, ++it) {
        data_[i] = *it;
    }
}

#if FASTINF_HAS_OPENCV
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(const cv::Mat& m) {
    constexpr int expected_depth = OpenCvTraits<_DType>::depth;
    if constexpr (_DType == DType::int64) {
        throw std::runtime_error(
            "cv::Mat to Tensor<int64> conversion is not supported");
    }

    if (m.depth() != expected_depth) {
        throw std::runtime_error("cv::Mat depth does not match Tensor dtype");
    }

    Shape shape;
    if (m.dims == 2) {
        shape = {1, static_cast<std::int64_t>(m.channels()),
                 static_cast<std::int64_t>(m.rows),
                 static_cast<std::int64_t>(m.cols)};
    } else {
        shape.reserve(static_cast<std::size_t>(m.dims) +
                      (m.channels() > 1 ? 1 : 0));
        for (int i = 0; i < m.dims; ++i) {
            shape.push_back(static_cast<std::int64_t>(m.size[i]));
        }
        if (m.channels() > 1) {
            shape.push_back(static_cast<std::int64_t>(m.channels()));
        }
    }

    desc_ = TensorDesc(shape, make_contiguous_strides(shape));
    const auto n = static_cast<std::size_t>(desc_.numel());
    data_ = new scalar_t[n];

    const cv::Mat src = m.isContinuous() ? m : m.clone();
    std::copy_n(reinterpret_cast<const scalar_t*>(src.data), n, data_);
}
#endif

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(const Tensor<_DType, _Device>& other)
    : desc_(other.desc_), quantization_(other.quantization_) {
    const auto n = static_cast<std::size_t>(desc_.numel());
    if (n == 0) {
        data_ = nullptr;
        return;
    }

    data_ = new scalar_t[n];
    std::copy(other.data_, other.data_ + n, data_);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator=(
    const Tensor& other) {
    if (this != &other) {
        Tensor tmp(other);
        std::swap(data_, tmp.data_);
        std::swap(desc_, tmp.desc_);
        std::swap(quantization_, tmp.quantization_);
    }

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      desc_(std::move(other.desc_)),
      quantization_(std::move(other.quantization_)) {
    const auto n = static_cast<std::size_t>(desc_.numel());
    other.data_ = nullptr;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator=(
    Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data_;

        data_ = other.data_;
        desc_ = std::move(other.desc_);
        quantization_ = std::move(other.quantization_);

        other.data_ = nullptr;
    }

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::~Tensor() {
    delete[] data_;
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::view_t Tensor<_DType, _Device>::view() const {
    return view_t(data_, desc_);
}

template <DType _DType, DeviceLikeType _Device>
const Shape& Tensor<_DType, _Device>::shape() const {
    return desc_.shape_;
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::scalar_t* Tensor<_DType, _Device>::data() {
    return data_;
}

template <DType _DType, DeviceLikeType _Device>
const typename Tensor<_DType, _Device>::scalar_t*
Tensor<_DType, _Device>::data() const {
    return data_;
}

template <DType _DType, DeviceLikeType _Device>
const TensorDesc& Tensor<_DType, _Device>::desc() const {
    return desc_;
}

template <DType _DType, DeviceLikeType _Device>
const std::optional<TensorQuantization>&
Tensor<_DType, _Device>::quantization() const {
    return quantization_;
}

template <DType _DType, DeviceLikeType _Device>
void Tensor<_DType, _Device>::set_quantization(
    TensorQuantization quantization) {
    quantization_ = quantization;
}

template <DType _DType, DeviceLikeType _Device>
void Tensor<_DType, _Device>::clear_quantization() {
    quantization_.reset();
}

template <DType _DType, DeviceLikeType _Device>
bool Tensor<_DType, _Device>::is_contiguous() const {
    return desc_.strides_ == make_contiguous_strides(desc_.shape_);
}

template <DType _DType, DeviceLikeType _Device>
int Tensor<_DType, _Device>::dim() const {
    return desc_.dim();
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::scalar_t& Tensor<_DType, _Device>::at(
    const Shape& indices) {
    return view().at(indices);
}

template <DType _DType, DeviceLikeType _Device>
const typename Tensor<_DType, _Device>::scalar_t& Tensor<_DType, _Device>::at(
    const Shape& indices) const {
    return view().at(indices);
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::view_t Tensor<_DType, _Device>::slice(
    const Shape& indices) const {
    return view().slice(indices);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor result(shape(), scalar_t{});
    auto it2 = result.begin();
    for (auto it1 = this->begin(); it1 != this->end(); ++it1, ++it2) {
        *it2 = *it1;
    }
    return result;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::clone() const {
    return Tensor(*this);
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::view_t Tensor<_DType, _Device>::transpose(
    std::size_t dim0, std::size_t dim1) const {
    return view().transpose(dim0, dim1);
}

template <DType _DType, DeviceLikeType _Device>
typename Tensor<_DType, _Device>::view_t Tensor<_DType, _Device>::t() const {
    return view().t();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator+=(
    const Tensor& other) {
    return *this += other.view();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator+=(
    const view_t& other) {
    view() += other;
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator+(
    const Tensor& other) const {
    return view() + other.view();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator+(
    const view_t& other) const {
    return view() + other;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator+=(scalar_t scalar) {
    std::for_each(begin(), end(), [scalar](auto& a) { a += scalar; });

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator+(
    scalar_t scalar) const {
    Tensor result = clone();
    result += scalar;
    return result;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator-=(
    const Tensor& other) {
    return *this -= other.view();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator-=(
    const view_t& other) {
    view() -= other;
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator-(
    const Tensor& other) const {
    return view() - other.view();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator-(
    const view_t& other) const {
    return view() - other;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator-=(scalar_t scalar) {
    std::for_each(begin(), end(), [scalar](auto& a) { a -= scalar; });

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator-(
    scalar_t scalar) const {
    Tensor result = clone();
    result -= scalar;
    return result;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator*=(
    const Tensor& other) {
    return *this *= other.view();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator*=(
    const view_t& other) {
    view() *= other;
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator*(
    const Tensor& other) const {
    return view() * other.view();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator*(
    const view_t& other) const {
    return view() * other;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>& Tensor<_DType, _Device>::operator*=(scalar_t scalar) {
    std::for_each(begin(), end(), [scalar](auto& a) { a *= scalar; });

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::operator*(
    scalar_t scalar) const {
    Tensor result = clone();
    result *= scalar;
    return result;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::mul(
    const Tensor& other) const {
    return view().mul(other.view());
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::mul(
    const view_t& other) const {
    return view().mul(other);
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator=(
    const Tensor<_DType, _Device>& other) {
    return *this = other.view();
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator+(const TensorView<_DType, _Device>& lhs,
                                  const TensorView<_DType, _Device>& rhs) {
    return add<_DType, _Device>(lhs, rhs);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator+(
    const TensorView<_DType, _Device>& input,
    typename TensorView<_DType, _Device>::scalar_t scalar) {
    return add<_DType, _Device>(input, scalar);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator+(
    typename TensorView<_DType, _Device>::scalar_t scalar,
    const TensorView<_DType, _Device>& input) {
    return add<_DType, _Device>(input, scalar);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator-(const TensorView<_DType, _Device>& lhs,
                                  const TensorView<_DType, _Device>& rhs) {
    return sub<_DType, _Device>(lhs, rhs);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator-(
    const TensorView<_DType, _Device>& input,
    typename TensorView<_DType, _Device>::scalar_t scalar) {
    return sub<_DType, _Device>(input, scalar);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator*(const TensorView<_DType, _Device>& lhs,
                                  const TensorView<_DType, _Device>& rhs) {
    return multiply<_DType, _Device>(lhs, rhs);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator*(
    const TensorView<_DType, _Device>& input,
    typename TensorView<_DType, _Device>::scalar_t scalar) {
    return multiply<_DType, _Device>(input, scalar);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> operator*(
    typename TensorView<_DType, _Device>::scalar_t scalar,
    const TensorView<_DType, _Device>& input) {
    return multiply<_DType, _Device>(input, scalar);
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> TensorView<_DType, _Device>::mul(
    const TensorView& other) const {
    return matmul<_DType, _Device>(*this, other);
}

template <DType _DType, DeviceLikeType _Device>
std::ostream& operator<<(std::ostream& os,
                         const Tensor<_DType, _Device>& tensor) {
    return TensorPrinter<_DType, _Device>::print(os, tensor.view());
}

};  // namespace fastinf
