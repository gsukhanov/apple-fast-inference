#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "fastinf/core/tensor.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
struct LayerState {
    using scalar_t = typename DTypeTraits<_DType>::type;
    using tensor_t = Tensor<_DType, _Device>;

    std::unordered_map<std::string, tensor_t> tensors;

    bool contains(std::string_view tensor_name) const;
    tensor_t find(std::string_view tensor_name) const;
};

template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
struct ModelLayer {
    using layer_state_t = LayerState<_DType, _Device>;

    std::unordered_map<std::string, layer_state_t> layers;

    bool contains(std::string_view layer_name) const;
    const layer_state_t& find(std::string_view layer_name) const;
};

enum class LoadLikeType : std::uint8_t { json };

template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu,
          LoadLikeType _Load = LoadLikeType::json>
class Loader;

template <DType _DType, DeviceLikeType _Device>
class Loader<_DType, _Device, LoadLikeType::json> {
 public:
    using model_layer_t = ModelLayer<_DType, _Device>;
    using layer_state_t = typename ModelLayer<_DType, _Device>::layer_state_t;
    using tensor_t = typename layer_state_t::tensor_t;
    using scalar_t = typename layer_state_t::scalar_t;

    model_layer_t load(std::string_view file_name);

 private:
    static void json2tensor(const nlohmann::json& node, Shape& shape,
                            std::vector<scalar_t>& data, std::size_t depth,
                            std::size_t& leaf_depth, bool& has_leaf_depth);
};

}  // namespace fastinf

#include "../../../src/core/serialization.hpp.inl"
