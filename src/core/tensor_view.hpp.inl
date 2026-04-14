#include "fastinf/core/tensor_view.hpp"
#include "fastinf/core/ops/common.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>::TensorView(
    typename TensorView<_DType, _Device>::scalar_t* data, TensorDesc desc)
    : data_(data), desc_(std::move(desc)) {
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>::TensorView(
    const Tensor<_DType, _Device>& tensor)
    : TensorView(tensor.data(), tensor.desc()) {
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::Iterator
TensorView<_DType, _Device>::begin() {
    return Iterator(data_, &desc_);
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::Iterator
TensorView<_DType, _Device>::end() {
    return Iterator(data_, &desc_, true);
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::ConstIterator
TensorView<_DType, _Device>::begin() const {
    return ConstIterator(data_, &desc_);
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::ConstIterator
TensorView<_DType, _Device>::end() const {
    return ConstIterator(data_, &desc_, true);
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::ConstIterator
TensorView<_DType, _Device>::cbegin() const {
    return begin();
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::ConstIterator
TensorView<_DType, _Device>::cend() const {
    return end();
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::scalar_t*
TensorView<_DType, _Device>::data() const {
    return data_ + desc_.offset_;
}

template <DType _DType, DeviceLikeType _Device>
const TensorDesc& TensorView<_DType, _Device>::desc() const {
    return desc_;
}

template <DType _DType, DeviceLikeType _Device>
const Shape& TensorView<_DType, _Device>::shape() const {
    return desc_.shape_;
}

template <DType _DType, DeviceLikeType _Device>
const Strides& TensorView<_DType, _Device>::strides() const {
    return desc_.strides_;
}

template <DType _DType, DeviceLikeType _Device>
std::int64_t TensorView<_DType, _Device>::numel() const {
    return desc_.numel();
}

template <DType _DType, DeviceLikeType _Device>
int TensorView<_DType, _Device>::dim() const {
    return desc_.dim();
}

template <DType _DType, DeviceLikeType _Device>
typename TensorView<_DType, _Device>::scalar_t& TensorView<_DType,
                                                            _Device>::at(
    const Shape& indices) {
    return data_[offset_for(indices)];
}

template <DType _DType, DeviceLikeType _Device>
const typename TensorView<_DType, _Device>::scalar_t&
TensorView<_DType, _Device>::at(const Shape& indices) const {
    return data_[offset_for(indices)];
}

template <DType _DType, DeviceLikeType _Device>
bool TensorView<_DType, _Device>::is_contiguous() const {
    return desc_.strides_ == make_contiguous_strides(desc_.shape_);
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator=(
    const TensorView& other) {
    check_same_shape(*this, other);

    auto it_dst = begin();
    auto it_src = other.begin();
    for (; it_dst != end(); ++it_dst, ++it_src) {
        *it_dst = *it_src;
    }

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator+=(
    const TensorView& other) {
    check_same_shape(*this, other);
    auto it_rhs = other.begin();
    for (auto it_lhs = begin(); it_lhs != end(); ++it_lhs, ++it_rhs) {
        *it_lhs += *it_rhs;
    }
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator+=(
    scalar_t scalar) {
    for (auto& value : *this) {
        value += scalar;
    }
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator-=(
    const TensorView& other) {
    check_same_shape(*this, other);
    auto it_rhs = other.begin();
    for (auto it_lhs = begin(); it_lhs != end(); ++it_lhs, ++it_rhs) {
        *it_lhs -= *it_rhs;
    }
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator-=(
    scalar_t scalar) {
    for (auto& value : *this) {
        value -= scalar;
    }
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator*=(
    const TensorView& other) {
    check_same_shape(*this, other);
    auto it_rhs = other.begin();
    for (auto it_lhs = begin(); it_lhs != end(); ++it_lhs, ++it_rhs) {
        *it_lhs *= *it_rhs;
    }
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>& TensorView<_DType, _Device>::operator*=(
    scalar_t scalar) {
    for (auto& value : *this) {
        value *= scalar;
    }
    return *this;
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device> TensorView<_DType, _Device>::slice(
    const Shape& indices) const {
    if (indices.size() > desc_.shape_.size()) {
        throw std::out_of_range("Too many indices for tensor slice");
    }

    TensorDesc tmp = desc_;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto index = indices[i];
        if (index < 0 || index >= desc_.shape_[i]) {
            throw std::out_of_range("Slice index out of range");
        }

        tmp.offset_ += index * desc_.strides_[i];
    }

    tmp.shape_.erase(
        tmp.shape_.begin(),
        tmp.shape_.begin() + static_cast<std::ptrdiff_t>(indices.size()));
    tmp.strides_.erase(
        tmp.strides_.begin(),
        tmp.strides_.begin() + static_cast<std::ptrdiff_t>(indices.size()));

    return TensorView(data_, std::move(tmp));
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device> TensorView<_DType, _Device>::permute(
    const Shape& perm) const {
    if (desc_.shape_.size() != perm.size()) {
        throw std::runtime_error(
            "permute(sparse_coo): number of dimensions in the tensor input "
            "does not match the length of the desired ordering of dimensions");
    }

    TensorDesc tmp;
    tmp.shape_.resize(perm.size());
    tmp.strides_.resize(perm.size());
    tmp.offset_ = desc_.offset_;

    std::vector<bool> used(perm.size(), false);

    for (std::size_t i = 0; i < perm.size(); ++i) {
        const auto dim = perm[i];

        if (dim < 0 || dim >= static_cast<std::int64_t>(desc_.shape_.size())) {
            throw std::out_of_range("Dimension out of range");
        }
        if (used[dim]) {
            throw std::runtime_error(
                "permute(): duplicate dims are not allowed");
        }

        tmp.shape_[i] = desc_.shape_[dim];
        tmp.strides_[i] = desc_.strides_[dim];
        used[dim] = true;
    }

    return TensorView(data_, std::move(tmp));
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device> TensorView<_DType, _Device>::transpose(
    std::size_t dim0, std::size_t dim1) const {
    if (dim0 >= desc_.shape_.size() || dim1 >= desc_.shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }

    if (dim0 == dim1) {
        return *this;
    }

    Shape order(desc_.shape_.size());
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[dim0], order[dim1]);

    return permute(order);
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device> TensorView<_DType, _Device>::t() const {
    if (desc_.dim() > 2) {
        throw std::out_of_range("Dimension out of range");
    }

    if (2 == desc_.dim()) {
        return transpose(0, 1);
    }

    return *this;
}

template <DType _DType, DeviceLikeType _Device>
std::size_t TensorView<_DType, _Device>::offset_for(
    const Shape& indices) const {
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
