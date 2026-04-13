#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "fastinf/nn/ops/softmax.hpp"
#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> softmax(const TensorView<_DType, _Device>& input,
                                int axis) {
    if (axis < 0 || axis >= input.dim()) {
        throw std::invalid_argument("Softmax axis out of range");
    }

    const auto& shape = input.shape();
    const auto axis_idx = static_cast<std::size_t>(axis);
    const auto axis_size = shape[axis_idx];
    if (axis_size == 0) {
        throw std::invalid_argument("Softmax axis must be non-empty");
    }

    Tensor<_DType, _Device> output(
        shape, typename Tensor<_DType, _Device>::scalar_t{});
    std::vector<std::int64_t> idx(shape.size());
    std::vector<std::size_t> dims_without_axis;
    dims_without_axis.reserve(shape.size());
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (d != axis_idx) {
            dims_without_axis.push_back(d);
        }
    }

    while (true) {
        typename Tensor<_DType, _Device>::scalar_t max_value =
            std::numeric_limits<
                typename Tensor<_DType, _Device>::scalar_t>::lowest();
        for (std::int64_t k = 0; k < axis_size; ++k) {
            idx[axis_idx] = k;
            max_value = std::max(max_value, input.at(idx));
        }

        double sum = 0.0;
        for (std::int64_t k = 0; k < axis_size; ++k) {
            idx[axis_idx] = k;
            const double exp_val =
                std::exp(static_cast<double>(input.at(idx) - max_value));
            output.at(idx) =
                static_cast<typename Tensor<_DType, _Device>::scalar_t>(exp_val);
            sum += exp_val;
        }

        if (!std::isfinite(sum) || sum == 0.0) {
            throw std::runtime_error(
                "Softmax encountered invalid normalization");
        }

        for (std::int64_t k = 0; k < axis_size; ++k) {
            idx[axis_idx] = k;
            output.at(idx) =
                static_cast<typename Tensor<_DType, _Device>::scalar_t>(
                    static_cast<double>(output.at(idx)) / sum);
        }

        idx[axis_idx] = 0;

        if (dims_without_axis.empty()) {
            break;
        }

        bool is_not_last_dim = false;
        for (auto it = dims_without_axis.rbegin();
             it != dims_without_axis.rend(); ++it) {
            const auto d = *it;
            ++idx[d];
            if (idx[d] < shape[d]) {
                is_not_last_dim = true;
                break;
            }
            idx[d] = 0;
        }

        if (!is_not_last_dim) {
            break;
        }
    }

    return output;
}
}  // namespace nn
}  // namespace fastinf
