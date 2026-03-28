#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "fastinf/core/device.hpp"
#include "fastinf/core/dtype.hpp"
#include "fastinf/core/serialization.hpp"
#include "fastinf/core/tensor.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class Layer {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;
    using layer_state_t = LayerState<_DType, _Device>;
    using model_layer_t = ModelLayer<_DType, _Device>;

    Layer() = default;
    Layer(const Layer&) = default;
    Layer& operator=(const Layer&) = default;
    virtual ~Layer() = default;

    tensor_t operator()(const tensor_t& input) const;

    virtual std::string name() const = 0;
    virtual void print(std::ostream& o, int level = 0) const;
    virtual tensor_t forward(const tensor_t& input) const = 0;
    virtual bool load_state(const layer_state_t& state);
    virtual bool load_state(const model_layer_t& state);

    void bind_identity(std::string key);
    const std::string& key() const;

 protected:
    template <typename... _Layers>
    void bind_children(_Layers&... layers);

 private:
    std::string key_;
    std::vector<Layer<_DType, _Device>*> children_;
};

template <DType _DType, DeviceLikeType _Device>
std::ostream& operator<<(std::ostream& o, const Layer<_DType, _Device>& m);

};  // namespace nn
}  // namespace fastinf

#include "../../../src/nn/layer.hpp.inl"
