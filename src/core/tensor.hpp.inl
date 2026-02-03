#include "fastinf/core/tensor.hpp"

namespace fastinf {
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
Tensor<_DType, _Device>::Tensor(const Tensor& other) : desc_(other.desc_) {
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
    }

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device>::Tensor(Tensor&& other) noexcept
    : data_(other.data_), desc_(std::move(other.desc_)) {
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
    return data_[offset_for(indices)];
}

template <DType _DType, DeviceLikeType _Device>
const typename Tensor<_DType, _Device>::scalar_t& Tensor<_DType, _Device>::at(
    const Shape& indices) const {
    return data_[offset_for(indices)];
}

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::clone() const {
    return Tensor(*this);
}

// Tensor& operator+=(const Tensor& other);
// Tensor operator+(const Tensor& other) const;
// Tensor& operator+=(scalar_t scalar);
// Tensor operator+(scalar_t scalar) const;

// Tensor& operator-=(const Tensor& other);
// Tensor operator-(const Tensor& other) const;
// Tensor& operator-=(scalar_t scalar);
// Tensor operator-(scalar_t scalar) const;

// Tensor& operator*=(const Tensor& other);
// Tensor operator*(const Tensor& other) const;
// Tensor& operator*=(scalar_t scalar);
// Tensor operator*(scalar_t scalar) const;

template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> Tensor<_DType, _Device>::mul(
    const Tensor& other) const {
    if (dim() == 1 && other.dim() == 1) {
        if (desc_.shape_[0] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b at "
                "non-singleton dimension 0");
        }
        scalar_t acc{};
        for (std::int64_t i = 0; i < desc_.shape_[0]; ++i) {
            acc += at({i}) * other.at({i});
        }

        return Tensor({}, std::vector<scalar_t>{acc});
    }

    if (dim() == 2 && other.dim() == 1) {
        if (desc_.shape_[1] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b at "
                "non-singleton dimension");
        }

        Tensor result({desc_.shape_[0]}, scalar_t{});
        for (std::int64_t i = 0; i < desc_.shape_[0]; ++i) {
            scalar_t acc{};
            for (std::int64_t k = 0; k < desc_.shape_[1]; ++k) {
                acc += at({i, k}) * other.at({k});
            }
            result.at({i}) = acc;
        }

        return result;
    }

    if (dim() == 1 && other.dim() == 2) {
        if (desc_.shape_[0] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b at "
                "non-singleton dimension");
        }
        Tensor result({other.desc_.shape_[1]}, scalar_t{});
        for (std::int64_t j = 0; j < other.desc_.shape_[1]; ++j) {
            scalar_t acc{};
            for (std::int64_t i = 0; i < desc_.shape_[0]; ++i) {
                acc += at({i}) * other.at({i, j});
            }
            result.at({j}) = acc;
        }

        return result;
    }

    if (dim() == 2 && other.dim() == 2) {
        if (desc_.shape_[1] != other.desc_.shape_[0]) {
            throw std::runtime_error(
                "The size of tensor a must match the size of tensor b at "
                "non-singleton dimension");
        }

        Tensor result({desc_.shape_[0], other.desc_.shape_[1]}, scalar_t{});
        for (std::int64_t i = 0; i < desc_.shape_[0]; ++i) {
            for (std::int64_t j = 0; j < other.desc_.shape_[1]; ++j) {
                scalar_t acc{};
                for (std::int64_t k = 0; k < desc_.shape_[1]; ++k) {
                    acc += at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = acc;
            }
        }

        return result;
    }

    throw std::runtime_error(
        "The size of tensor a and tensor b must be less than 3");
}

// template <>
// Tensor<DType::float32, DeviceLikeType::neon>
// Tensor<DType::float32, DeviceLikeType::neon>::mul(const Tensor& other) const
// {
// }

// template <>
// Tensor<DType::float32, DeviceLikeType::amx>
// Tensor<DType::float32, DeviceLikeType::amx>::mul(const Tensor& other) const {
// }

template <DType _DType, DeviceLikeType _Device>
std::size_t Tensor<_DType, _Device>::offset_for(const Shape& indices) const {
    if (indices.size() != desc_.shape_.size()) {
        throw std::out_of_range("too many indices for tensor of dimension");
    }
    size_t offset = desc_.offset_;
    for (size_t i = 0; i < desc_.shape_.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= desc_.shape_[i]) {
            throw std::out_of_range("index is out of bounds for dimension");
        }
        offset += indices[i] * desc_.strides_[i];
    }
    return offset;
}

};  // namespace fastinf
