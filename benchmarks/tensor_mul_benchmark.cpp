#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "fastinf/core.hpp"

using namespace fastinf;

namespace {
template <typename T>
std::vector<T> makeMatrixData(std::int64_t rows, std::int64_t cols, T scale) {
    std::vector<T> data(static_cast<std::size_t>(rows * cols));

    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            const auto index = static_cast<std::size_t>(i * cols + j);
            data[index] = static_cast<T>(((i + 1) * (j + 3)) % 17 - 8) * scale;
        }
    }

    return data;
}

template <DType DTYPE, DeviceLikeType DEVICE>
void BM_TensorMul(benchmark::State& state) {
    using TensorType = Tensor<DTYPE, DEVICE>;
    using Scalar = typename TensorType::scalar_t;

    const auto m = state.range(0);
    const auto k = state.range(1);
    const auto n = state.range(2);

    const TensorType lhs({m, k}, makeMatrixData<Scalar>(m, k, Scalar{0.25}));
    const TensorType rhs({k, n}, makeMatrixData<Scalar>(k, n, Scalar{0.5}));

    for (auto _ : state) {
        const auto result = lhs.mul(rhs);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * k * n);
}
}  // namespace

BENCHMARK_TEMPLATE(BM_TensorMul, DType::float32, DeviceLikeType::cpu)
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({128, 256, 64});

BENCHMARK_TEMPLATE(BM_TensorMul, DType::float64, DeviceLikeType::cpu)
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({128, 256, 64});

BENCHMARK_TEMPLATE(BM_TensorMul, DType::float32, DeviceLikeType::amx)
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({128, 256, 64});

BENCHMARK_TEMPLATE(BM_TensorMul, DType::float64, DeviceLikeType::amx)
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({128, 256, 64});
