#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

#include "fastinf/core.hpp"
#include "fastinf/nn.hpp"
#include "fastinf/nn/backend/neon/conv2d.hpp"

using namespace fastinf;
using namespace fastinf::nn;

namespace {
template <template <DType, DeviceLikeType> class _ConvLayer, DType _DType,
          DeviceLikeType _Device>
class BenchmarkLeNet : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;

    BenchmarkLeNet(std::int64_t num_classes = 10)
        : conv1(1, 6, 5),
          relu1(),
          pool1(2, 2),
          conv2(6, 16, 5),
          relu2(),
          pool2(2, 2),
          flatten(),
          fc1(16 * 4 * 4, 120),
          relu3(),
          fc2(120, 84),
          relu4(),
          fc3(84, num_classes) {
        conv1.bind_identity("conv1");
        conv2.bind_identity("conv2");
        fc1.bind_identity("fc1");
        fc2.bind_identity("fc2");
        fc3.bind_identity("fc3");
        this->bind_children(conv1, conv2, fc1, fc2, fc3);
    }

    std::string name() const override {
        return "nn.BenchmarkLeNet";
    }

    tensor_t forward(const tensor_t& input) const override {
        tensor_t output = input;
        output = pool1(relu1(conv1(output)));
        output = pool2(relu2(conv2(output)));

        output = flatten(output);
        output = relu3(fc1(output));
        output = relu4(fc2(output));
        output = fc3(output);

        return output;
    }

 private:
    _ConvLayer<_DType, _Device> conv1, conv2;
    ReLU<_DType, _Device> relu1, relu2, relu3, relu4;
    AvgPool2d<_DType, _Device> pool1, pool2;
    Flatten<_DType, _Device> flatten;
    Linear<_DType, _Device> fc1, fc2, fc3;
};

template <DType _DType, DeviceLikeType _Device>
using GreedyLeNet = BenchmarkLeNet<Conv2d, _DType, _Device>;

template <DType _DType, DeviceLikeType _Device>
using Im2ColLeNet = BenchmarkLeNet<ConvIm2Col, _DType, _Device>;

template <typename TensorType>
TensorType makeInput(std::int64_t batch_size) {
    using scalar_t = typename TensorType::scalar_t;

    TensorType input({batch_size, 1, 28, 28});
    std::int64_t index = 0;
    for (auto& value : input) {
        value = static_cast<scalar_t>((index % 251) / 255.0);
        ++index;
    }
    return input;
}

template <typename Model>
void runLeNetBenchmark(benchmark::State& state) {
    using tensor_t = typename Model::tensor_t;

    const auto batch_size = state.range(0);
    const Model model;
    const auto input = makeInput<tensor_t>(batch_size);

    for (auto _ : state) {
        const auto output = model(input);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("batch=" + std::to_string(batch_size));
}

static void BM_LeNetFloat32Neon(benchmark::State& state) {
    runLeNetBenchmark<Im2ColLeNet<DType::float32, DeviceLikeType::neon>>(state);
}

static void BM_LeNetFloat32CPU(benchmark::State& state) {
    runLeNetBenchmark<GreedyLeNet<DType::float32, DeviceLikeType::cpu>>(state);
}

#if FASTINF_HAS_ACCELERATE
static void BM_LeNetFloat32AMX(benchmark::State& state) {
    runLeNetBenchmark<GreedyLeNet<DType::float32, DeviceLikeType::amx>>(state);
}
#endif
}  // namespace

BENCHMARK(BM_LeNetFloat32Neon)->Arg(1)->Arg(5)->Arg(10);
BENCHMARK(BM_LeNetFloat32CPU)->Arg(1)->Arg(5)->Arg(10);
#if FASTINF_HAS_ACCELERATE
BENCHMARK(BM_LeNetFloat32AMX)->Arg(1)->Arg(5)->Arg(10);
#endif
