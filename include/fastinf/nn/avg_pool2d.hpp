#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "layer.hpp"
#include "ops.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class AvgPool2d : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;
    struct size_2_t {
        std::int64_t h, w;
    };

    AvgPool2d(size_2_t kernel, size_2_t stride = {1, 1},
              size_2_t padding = {0, 0});

    AvgPool2d(std::int64_t kernel, std::int64_t stride = 1,
              std::int64_t padding = 0);

    std::string name() const override;
    tensor_t forward(const tensor_t& input) const override;

 private:
    size_2_t kernel_;
    size_2_t stride_;
    size_2_t padding_;

    std::int64_t output_height(std::int64_t input_h) const;
    std::int64_t output_width(std::int64_t input_w) const;
};
}  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/avg_pool2d.hpp.inl"
