#pragma once

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

#include "device.hpp"
#include "dtype.hpp"
#include "tensor_desc.hpp"
#include "tensor_view.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class TensorPrinter {
 public:
    using view_t = TensorView<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    static std::ostream& print(std::ostream& os, const view_t& view);

 private:
    static std::string format_scalar(const scalar_t& value);

    template <class Iterator>
    static void print_nested(std::ostream& os, const Shape& shape,
                             std::size_t dim, std::size_t indent, Iterator& it);
};

};  // namespace fastinf

#include "../../../src/core/tensor_printer.hpp.inl"
