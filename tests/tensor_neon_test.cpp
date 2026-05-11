#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#if FASTINF_HAS_NEON
#include "fastinf/core/backend/neon/matmul.hpp"
#include "fastinf/nn/backend/neon/conv2d.hpp"

using namespace fastinf;
using namespace fastinf::nn;

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

TEST(NeonConv2d, Im2ColMatchesCpuConv2dWithBias) {
    Tensor<DType::float32, DeviceLikeType::cpu> cpu_input(
        {1, 2, 4, 4},
        {
            1.0F,  2.0F,  3.0F,  4.0F,
            5.0F,  6.0F,  7.0F,  8.0F,
            9.0F,  10.0F, 11.0F, 12.0F,
            13.0F, 14.0F, 15.0F, 16.0F,
            -1.0F, -2.0F, -3.0F, -4.0F,
            -5.0F, -6.0F, -7.0F, -8.0F,
            -9.0F, -10.0F, -11.0F, -12.0F,
            -13.0F, -14.0F, -15.0F, -16.0F,
        });
    Tensor<DType::float32, DeviceLikeType::cpu> cpu_weight(
        {3, 2, 3, 3},
        {
            0.25F, 0.0F, -0.25F, 0.5F, 0.0F, -0.5F, 0.25F, 0.0F, -0.25F,
            0.1F,  0.2F, 0.3F,   0.0F, 0.1F, 0.2F,  0.3F,  0.0F, 0.1F,
            -0.3F, 0.2F, 0.1F,   0.4F, 0.0F, -0.4F, 0.1F,  0.2F, -0.3F,
            0.2F,  -0.2F, 0.2F,  -0.2F, 0.2F, -0.2F, 0.2F,  -0.2F, 0.2F,
            0.05F, 0.1F, 0.15F,  0.2F, 0.25F, 0.3F, 0.35F, 0.4F, 0.45F,
            -0.1F, -0.2F, -0.3F, 0.3F, 0.2F, 0.1F,  -0.1F, 0.0F, 0.1F,
        });
    Tensor<DType::float32, DeviceLikeType::cpu> cpu_bias({3},
                                                        {0.5F, -1.0F, 2.0F});

    Tensor<DType::float32, DeviceLikeType::neon> neon_input(
        cpu_input.shape(), std::vector<float>(cpu_input.begin(), cpu_input.end()));
    Tensor<DType::float32, DeviceLikeType::neon> neon_weight(
        cpu_weight.shape(),
        std::vector<float>(cpu_weight.begin(), cpu_weight.end()));
    Tensor<DType::float32, DeviceLikeType::neon> neon_bias(
        cpu_bias.shape(), std::vector<float>(cpu_bias.begin(), cpu_bias.end()));

    const auto cpu_bias_view = cpu_bias.view();
    const auto neon_bias_view = neon_bias.view();
    auto cpu_result = conv2d<DType::float32, DeviceLikeType::cpu>(
        cpu_input.view(), cpu_weight.view(), {2, 1}, {1, 1}, {1, 1}, 1,
        std::cref(cpu_bias_view));
    auto neon_result = conv2d<DType::float32, DeviceLikeType::neon>(
        neon_input.view(), neon_weight.view(), {2, 1}, {1, 1}, {1, 1}, 1,
        std::cref(neon_bias_view));

    ASSERT_EQ(neon_result.shape(), cpu_result.shape());
    for (std::int64_t i = 0; i < cpu_result.desc().numel(); ++i) {
        EXPECT_NEAR(neon_result.data()[i], cpu_result.data()[i], 1e-5F);
    }
}
#endif
