#include <gtest/gtest.h>

#include "fastinf/core.hpp"

using namespace fastinf;

namespace {
using AmxFloatTensor = Tensor<DType::float32, DeviceLikeType::amx>;

void expectShape(const AmxFloatTensor& tensor, const Shape& expected) {
    ASSERT_EQ(tensor.shape(), expected);
}

void expectElementsNear(const AmxFloatTensor& tensor,
                        const std::vector<float>& expected,
                        float tolerance = 1e-5f) {
    ASSERT_EQ(static_cast<std::size_t>(tensor.desc().numel()), expected.size());

    std::size_t index = 0;
    for (const auto value : tensor) {
        EXPECT_NEAR(value, expected[index], tolerance);
        ++index;
    }
}
}  // namespace

TEST(TensorAmx, MulSquareMatrices) {
    const AmxFloatTensor lhs({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const AmxFloatTensor rhs({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    const auto result = lhs.mul(rhs);

    expectShape(result, {2, 2});
    expectElementsNear(result, {58.0f, 64.0f, 139.0f, 154.0f});
}

TEST(TensorAmx, MulRectangularMatrices) {
    const AmxFloatTensor lhs({3, 2}, {1.5f, -2.0f, 0.0f, 4.0f, -1.0f, 3.0f});
    const AmxFloatTensor rhs(
        {2, 4}, {2.0f, -1.0f, 0.5f, 3.0f, -4.0f, 2.0f, 1.5f, -2.0f});

    const auto result = lhs.mul(rhs);

    expectShape(result, {3, 4});
    expectElementsNear(result, {11.0f, -5.5f, -2.25f, 8.5f, -16.0f, 8.0f, 6.0f,
                                -8.0f, -14.0f, 7.0f, 4.0f, -9.0f});
}

TEST(TensorAmx, MulThrowsOnIncompatibleShapes) {
    const AmxFloatTensor lhs({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const AmxFloatTensor rhs({2, 2}, {7.0f, 8.0f, 9.0f, 10.0f});

    EXPECT_THROW(lhs.mul(rhs), std::runtime_error);
}
