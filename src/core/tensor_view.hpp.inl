#include "fastinf/core/tensor_view.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device>::TensorView(
    typename TensorView<_DType, _Device>::scalar_t* data, TensorDesc desc)
    : data_(data), desc_(std::move(desc)) {
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
bool TensorView<_DType, _Device>::is_contiguous() const {
    return desc_.strides_ == make_contiguous_strides(desc_.shape_);
}

template <DType _DType, DeviceLikeType _Device>
TensorView<_DType, _Device> TensorView<_DType, _Device>::permute(
    const Shape& perm) {
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

        if (dim < 0 ||
            dim >= static_cast<std::int64_t>(desc_.shape_.size())) {
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

};  // namespace fastinf
