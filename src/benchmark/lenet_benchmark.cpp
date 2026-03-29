#include <benchmark/benchmark.h>

#include "fastinf/core.hpp"
#include "fastinf/nn.hpp"

using namespace fastinf;
using namespace fastinf::nn;

template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class LeNet : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    LeNet(std::int64_t num_classes = 10)
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
        return "nn.LeNet";
    }

    tensor_t forward(const tensor_t& x) const override {
        tensor_t out = x;
        out = pool1(relu1(conv1(out)));
        out = pool2(relu2(conv2(out)));

        out = flatten(out);
        out = relu3(fc1(out));
        out = relu4(fc2(out));
        out = fc3(out);

        return out;
    }

 private:
    Conv2d<_DType, _Device> conv1, conv2;
    ReLU<_DType, _Device> relu1, relu2, relu3, relu4;
    AvgPool2d<_DType, _Device> pool1, pool2;
    Flatten<_DType, _Device> flatten;
    Linear<_DType, _Device> fc1, fc2, fc3;
};

static void BM_LeNetFloat32CPU(benchmark::State& state) {
    LeNet<DType::float32, DeviceLikeType::cpu> model;

    Tensor<DType::float32, DeviceLikeType::cpu> tensor({1, 1, 28, 28}, 1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(model(tensor));
    }
}

static void BM_LeNetFloat32CPUBatch5(benchmark::State& state) {
    LeNet<DType::float64, DeviceLikeType::cpu> model;

    Tensor<DType::float64, DeviceLikeType::cpu> tensor({5, 1, 28, 28}, 1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(model(tensor));
    }
}

BENCHMARK(BM_LeNetFloat32CPU);
BENCHMARK(BM_LeNetFloat32CPUBatch5);