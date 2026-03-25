#include "fastinf/core/tensor_iterator.hpp"

namespace fastinf {
template <class T>
TensorIterator<T>::TensorIterator(pointer data,
                                  const TensorDesc* desc,
                                  bool end)
    : m_data(data), m_desc(desc) {
    if (m_desc == nullptr) {
        return;
    }

    m_numel = m_desc->numel();
    m_linear_index = end ? m_numel : 0;

    m_is_contiguous =
        (m_desc->strides_ == make_contiguous_strides(m_desc->shape_));

    if (m_is_contiguous) {
        m_offset = m_desc->offset_ + m_linear_index;
        return;
    }

    if (m_linear_index >= m_numel) {
        return;
    }

    m_offset = m_desc->offset_;
    m_indices.assign(m_desc->shape_.size(), 0);
}

template <class T>
typename TensorIterator<T>::reference TensorIterator<T>::operator*() const {
    return m_data[m_offset];
}

template <class T>
typename TensorIterator<T>::pointer TensorIterator<T>::operator->() const {
    return &m_data[m_offset];
}

template <class T>
TensorIterator<T>& TensorIterator<T>::operator++() {
    if (m_desc == nullptr || m_linear_index >= m_numel) {
        return *this;
    }

    ++m_linear_index;

    if (m_is_contiguous) {
        ++m_offset;
        return *this;
    }

    if (m_linear_index >= m_numel || m_desc->shape_.empty()) {
        return *this;
    }

    for (std::size_t d = m_desc->shape_.size(); d-- > 0;) {
        ++m_indices[d];
        m_offset += m_desc->strides_[d];

        if (m_indices[d] < m_desc->shape_[d]) {
            break;
        }

        m_indices[d] = 0;
        m_offset -= m_desc->shape_[d] * m_desc->strides_[d];
    }

    return *this;
}

template <class T>
TensorIterator<T> TensorIterator<T>::operator++(int) {
    TensorIterator tmp = *this;
    ++(*this);
    return tmp;
}

template <class T>
bool TensorIterator<T>::operator==(const TensorIterator& other) const {
    return m_data == other.m_data && m_desc == other.m_desc &&
           m_linear_index == other.m_linear_index;
}

template <class T>
bool TensorIterator<T>::operator!=(const TensorIterator& other) const {
    return !(*this == other);
}

};  // namespace fastinf
