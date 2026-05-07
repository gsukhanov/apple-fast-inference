#include <benchmark/benchmark.h>

#include "fastinf/core.hpp"
#include "fastinf/nn.hpp"

using namespace fastinf;
using namespace fastinf::nn;

static void BM_LeNetFloat32CPU(benchmark::State& state) {
    LeNet<DType::float32, DeviceLikeType::cpu> model;

    Tensor<DType::float32, DeviceLikeType::cpu> tensor({1, 1, 28, 28}, 1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(model(tensor));
    }
}

static void BM_LeNetFloat32CPUBatch5(benchmark::State& state) {
    LeNet<DType::float32, DeviceLikeType::cpu> model;

    Tensor<DType::float32, DeviceLikeType::cpu> tensor({5, 1, 28, 28}, 1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(model(tensor));
    }
}

BENCHMARK(BM_LeNetFloat32CPU);
BENCHMARK(BM_LeNetFloat32CPUBatch5);