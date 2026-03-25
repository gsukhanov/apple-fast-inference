#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <vector>

#include "tensor_desc.hpp"

namespace fastinf {
template <class T>
class TensorIterator {
 public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_const_t<T>;
    using pointer = T*;
    using reference = T&;

    TensorIterator() = default;
    TensorIterator(pointer data, const TensorDesc* desc, bool end = false);

    reference operator*() const;
    pointer operator->() const;

    TensorIterator& operator++();
    TensorIterator operator++(int);

    bool operator==(const TensorIterator& other) const;
    bool operator!=(const TensorIterator& other) const;

 private:
    pointer m_data{nullptr};
    const TensorDesc* m_desc{nullptr};
    std::int64_t m_numel{0};
    std::int64_t m_linear_index{0};
    std::int64_t m_offset{0};
    bool m_is_contiguous{true};
    std::vector<std::int64_t> m_indices;
};

};  // namespace fastinf

#include "../../../src/core/tensor_iterator.hpp.inl"
