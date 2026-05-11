#include "fastinf/core/backend/neon/blocking.hpp"

#include <algorithm>
#include <stdexcept>

namespace fastinf {
namespace {
inline std::int64_t div_ceil(std::int64_t value, std::int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

template<DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> pack_2d(const TensorView<_DType, _Device>& input,
                                std::int64_t row_block,
                                std::int64_t col_block) {
    using scalar_t = typename DTypeTraits<_DType>::type;

    if (input.dim() != 2) {
        throw std::runtime_error("Only two-dimensional tensors can be packed");
    }

    const std::int64_t rows = input.shape()[0];
    const std::int64_t cols = input.shape()[1];
    const std::int64_t block_rows = div_ceil(rows, row_block);
    const std::int64_t block_cols = div_ceil(cols, col_block);

    Tensor<_DType, _Device> packed(
        Shape{block_rows, block_cols, row_block, col_block}, scalar_t{0});

    for (std::int64_t br = 0; br < block_rows; ++br) {
        for (std::int64_t bc = 0; bc < block_cols; ++bc) {
            for (std::int64_t r = 0; r < row_block; ++r) {
                const std::int64_t src_r = br * row_block + r;
                if (src_r >= rows) {
                    continue;
                }

                for (std::int64_t c = 0; c < col_block; ++c) {
                    const std::int64_t src_c = bc * col_block + c;
                    if (src_c < cols) {
                        packed.at(Shape{br, bc, r, c}) =
                            input.at(Shape{src_r, src_c});
                    }
                }
            }
        }
    }

    return packed;
}

template<DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> unpack_2d(const Tensor<_DType, _Device>& input,
                                  const Shape& output_shape,
                                  std::int64_t row_block,
                                  std::int64_t col_block) {
    using scalar_t = typename DTypeTraits<_DType>::type;

    if (input.dim() != 4 || output_shape.size() != 2) {
        throw std::runtime_error(
            "Blocked tensor must be 4D and output shape must be 2D");
    }

    Tensor<_DType, _Device> unpacked(output_shape, scalar_t{0});
    const std::int64_t rows = output_shape[0];
    const std::int64_t cols = output_shape[1];

    for (std::int64_t br = 0; br < input.shape()[0]; ++br) {
        for (std::int64_t bc = 0; bc < input.shape()[1]; ++bc) {
            for (std::int64_t r = 0; r < row_block; ++r) {
                const std::int64_t dst_r = br * row_block + r;
                if (dst_r >= rows) {
                    continue;
                }

                for (std::int64_t c = 0; c < col_block; ++c) {
                    const std::int64_t dst_c = bc * col_block + c;
                    if (dst_c < cols) {
                        unpacked.at(Shape{dst_r, dst_c}) =
                            input.at(Shape{br, bc, r, c});
                    }
                }
            }
        }
    }

    if (input.quantization().has_value()) {
        unpacked.set_quantization(*input.quantization());
    }
    return unpacked;
}
}  // namespace

template<DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> pack_input(const Tensor<_DType, _Device>& input) {
    auto packed = pack_input(input.view());
    if (input.quantization().has_value()) {
        packed.set_quantization(*input.quantization());
    }
    return packed;
}

template<DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> pack_input(const TensorView<_DType, _Device>& input) {
    if (input.dim() == 1) {
        Tensor<_DType, _Device> matrix(Shape{1, input.shape()[0]});
        for (std::int64_t i = 0; i < input.shape()[0]; ++i) {
            matrix.at(Shape{0, i}) = input.at(Shape{i});
        }
        return pack_2d(matrix.view(), KERNEL_RESULT_WIDTH, KERNEL_DEPTH);
    }

    return pack_2d(input, KERNEL_RESULT_WIDTH, KERNEL_DEPTH);
}

template<DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> pack_result(const Tensor<_DType, _Device>& input) {
    auto packed = pack_result(input.view());
    if (input.quantization().has_value()) {
        packed.set_quantization(*input.quantization());
    }
    return packed;
}

template<DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> pack_result(const TensorView<_DType, _Device>& input) {
    if (input.dim() == 1) {
        Tensor<_DType, _Device> matrix(Shape{input.shape()[0], 1});
        for (std::int64_t i = 0; i < input.shape()[0]; ++i) {
            matrix.at(Shape{i, 0}) = input.at(Shape{i});
        }
        return pack_2d(matrix.view(), KERNEL_RESULT_WIDTH,
                       KERNEL_RESULT_WIDTH);
    }

    return pack_2d(input, KERNEL_RESULT_WIDTH, KERNEL_RESULT_WIDTH);
}

template<DType _DType, DeviceLikeType _Device>
Tensor<_DType, _Device> unpack_result(const Tensor<_DType, _Device>& input,
                                      const Shape& output_shape) {
    if (output_shape.size() == 1) {
        auto matrix = unpack_2d(input, Shape{output_shape[0], 1},
                                KERNEL_RESULT_WIDTH, KERNEL_RESULT_WIDTH);
        Tensor<_DType, _Device> vector(output_shape);
        for (std::int64_t i = 0; i < output_shape[0]; ++i) {
            vector.at(Shape{i}) = matrix.at(Shape{i, 0});
        }
        if (matrix.quantization().has_value()) {
            vector.set_quantization(*matrix.quantization());
        }
        return vector;
    }

    return unpack_2d(input, output_shape, KERNEL_RESULT_WIDTH,
                     KERNEL_RESULT_WIDTH);
}

}  // namespace fastinf
