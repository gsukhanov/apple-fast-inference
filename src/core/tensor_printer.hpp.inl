#include "fastinf/core/tensor_printer.hpp"

#include <sstream>
#include <type_traits>


namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
std::string TensorPrinter<_DType, _Device>::format_scalar(
    const scalar_t& value) {
    using scalar_type = std::remove_cv_t<scalar_t>;

    std::ostringstream oss;
    if constexpr (std::is_same_v<scalar_type, std::int8_t> ||
                  std::is_same_v<scalar_type, std::uint8_t>) {
        oss << static_cast<int>(value);
        return oss.str();
    }

    if constexpr (std::is_floating_point_v<scalar_type>) {
        oss << value;
        std::string s = oss.str();
        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
            s.find('E') == std::string::npos) {
            s.push_back('.');
        }
        return s;
    }

    oss << value;
    return oss.str();
}

template <DType _DType, DeviceLikeType _Device>
template <class Iterator>
void TensorPrinter<_DType, _Device>::print_nested(std::ostream& os,
                                                   const Shape& shape,
                                                   std::size_t dim,
                                                   std::size_t indent,
                                                   Iterator& it) {
    if (dim == shape.size()) {
        os << format_scalar(*it);
        ++it;
        return;
    }

    os << "[";
    const auto size = shape[dim];
    if (size == 0) {
        os << "]";
        return;
    }

    for (std::int64_t i = 0; i < size; ++i) {
        if (i > 0) {
            os << ",";
            if (dim + 1 == shape.size()) {
                os << " ";
            } else if (shape.size() - dim > 2) {
                os << "\n\n" << std::string(indent + 1, ' ');
            } else {
                os << "\n" << std::string(indent + 1, ' ');
            }
        }

        print_nested(os, shape, dim + 1, indent + 1, it);
    }

    os << "]";
}

template <DType _DType, DeviceLikeType _Device>
std::ostream& TensorPrinter<_DType, _Device>::print(std::ostream& os,
                                                     const view_t& view) {
    os << "tensor(";

    const auto& shape = view.shape();
    auto it = view.cbegin();

    if (shape.empty()) {
        os << format_scalar(*it) << ")";
        return os;
    }

    print_nested(os, shape, 0, 7, it);
    os << ")";
    return os;
}

};  // namespace fastinf
