#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#if FASTINF_HAS_NEON
#include "fastinf/core/backend/neon/matmul.hpp"

using namespace fastinf;

namespace {
using NeonI8Tensor = Tensor<DType::int8, DeviceLikeType::neon>;
using Scalar = typename DTypeTraits<DType::int8>::type;

constexpr float kScale = 1.0F / DTypeTraits<DType::int8>::max;

Scalar requantize(std::int32_t value) {
    const auto scaled = std::llround(value * kScale * kScale *
                                    DTypeTraits<DType::int8>::max);
    const auto clamped = std::clamp<long long>(
        scaled, static_cast<long long>(std::numeric_limits<Scalar>::lowest()),
        static_cast<long long>(DTypeTraits<DType::int8>::max));
    return static_cast<Scalar>(clamped);
}

NeonI8Tensor make_tensor(const Shape& shape, const std::vector<Scalar>& data) {
    return NeonI8Tensor(shape, data);
}

QuantizeDesc<DType::int8, DType::float32> qdesc() {
    return QuantizeDesc<DType::int8, DType::float32>(0, kScale);
}
}  // namespace

TEST(NeonMatmul, VectorVector) {
    auto lhs = make_tensor({4}, {32, 32, 32, 32});
    auto rhs = make_tensor({4}, {32, 32, 32, 32});

    auto result =
        matmul<DType::int8, DType::float32>(lhs.view(), rhs.view(), qdesc(),
                                            qdesc());

    ASSERT_EQ(result.dim(), 0);
    EXPECT_EQ(result.at({}), requantize(4 * 32 * 32));
}

TEST(NeonMatmul, MatrixVectorWithPadding) {
    auto lhs = make_tensor({3, 5}, {
                                       10, 20, 30, 40, 50,
                                       5,  6,  7,  8,  9,
                                       32, 32, 32, 32, 32,
                                   });
    auto rhs = make_tensor({5}, {3, 4, 5, 6, 7});

    auto result =
        matmul<DType::int8, DType::float32>(lhs.view(), rhs.view(), qdesc(),
                                            qdesc());

    ASSERT_EQ(result.shape(), (Shape{3}));
    EXPECT_EQ(result.at({0}), requantize(10 * 3 + 20 * 4 + 30 * 5 + 40 * 6 +
                                         50 * 7));
    EXPECT_EQ(result.at({1}), requantize(5 * 3 + 6 * 4 + 7 * 5 + 8 * 6 +
                                         9 * 7));
    EXPECT_EQ(result.at({2}), requantize(32 * 3 + 32 * 4 + 32 * 5 + 32 * 6 +
                                         32 * 7));
}

TEST(NeonMatmul, VectorMatrixWithPadding) {
    auto lhs = make_tensor({5}, {8, 9, 10, 11, 12});
    auto rhs = make_tensor({5, 4}, {
                                       1, 2, 3, 4,
                                       5, 6, 7, 8,
                                       9, 10, 11, 12,
                                       13, 14, 15, 16,
                                       17, 18, 19, 20,
                                   });

    auto result =
        matmul<DType::int8, DType::float32>(lhs.view(), rhs.view(), qdesc(),
                                            qdesc());

    ASSERT_EQ(result.shape(), (Shape{4}));
    EXPECT_EQ(result.at({0}), requantize(8 * 1 + 9 * 5 + 10 * 9 + 11 * 13 +
                                         12 * 17));
    EXPECT_EQ(result.at({1}), requantize(8 * 2 + 9 * 6 + 10 * 10 + 11 * 14 +
                                         12 * 18));
    EXPECT_EQ(result.at({2}), requantize(8 * 3 + 9 * 7 + 10 * 11 + 11 * 15 +
                                         12 * 19));
    EXPECT_EQ(result.at({3}), requantize(8 * 4 + 9 * 8 + 10 * 12 + 11 * 16 +
                                         12 * 20));
}

TEST(NeonMatmul, MatrixMatrixAcrossKernelBlocks) {
    auto lhs = make_tensor({10, 5}, {
                                        1,  2,  3,  4,  5,
                                        6,  7,  8,  9,  10,
                                        11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35,
                                        36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45,
                                        46, 47, 48, 49, 50,
                                    });
    auto rhs = make_tensor({5, 10}, {
                                        1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                    });

    auto result =
        matmul<DType::int8, DType::float32>(lhs.view(), rhs.view(), qdesc(),
                                            qdesc());

    ASSERT_EQ(result.shape(), (Shape{10, 10}));
    for (std::int64_t i = 0; i < 10; ++i) {
        for (std::int64_t j = 0; j < 10; ++j) {
            std::int32_t expected = 0;
            for (std::int64_t k = 0; k < 5; ++k) {
                expected += lhs.at({i, k}) * rhs.at({k, j});
            }
            EXPECT_EQ(result.at({i, j}), requantize(expected));
        }
    }
}
#endif
