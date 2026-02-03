#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace fastinf {
template <class T>
class TensorIterator {
 public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_const_t<T>;
    using pointer = T*;
    using reference = T&;

    explicit TensorIterator(pointer ptr) : m_ptr(ptr) {
    }

    reference operator*() const {
        return *m_ptr;
    }
    pointer operator->() const {
        return m_ptr;
    }

    TensorIterator& operator++() {
        ++m_ptr;
        return *this;
    }

    TensorIterator operator++(int) {
        TensorIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const TensorIterator& a, const TensorIterator& b) {
        return a.m_ptr == b.m_ptr;
    }
    friend bool operator!=(const TensorIterator& a, const TensorIterator& b) {
        return a.m_ptr != b.m_ptr;
    }

 private:
    pointer m_ptr;
};
};  // namespace fastinf
