#include "fastinf/core/serialization.hpp"

namespace fastinf {
template <DType _DType, DeviceLikeType _Device>
bool LayerState<_DType, _Device>::contains(std::string_view tensor_name) const {
    return tensors.find(std::string(tensor_name)) != tensors.end();
}

template <DType _DType, DeviceLikeType _Device>
typename LayerState<_DType, _Device>::tensor_t
LayerState<_DType, _Device>::find(std::string_view tensor_name) const {
    return tensors.at(std::string(tensor_name));
}

template <DType _DType, DeviceLikeType _Device>
bool ModelLayer<_DType, _Device>::contains(std::string_view layer_name) const {
    return layers.find(std::string(layer_name)) != layers.end();
}

template <DType _DType, DeviceLikeType _Device>
const typename ModelLayer<_DType, _Device>::layer_state_t&
ModelLayer<_DType, _Device>::find(std::string_view layer_name) const {
    return layers.at(std::string(layer_name));
}

template <DType _DType, DeviceLikeType _Device>
typename Loader<_DType, _Device, LoadLikeType::json>::model_layer_t
Loader<_DType, _Device, LoadLikeType::json>::load(std::string_view file_name) {
    std::ifstream input{std::string(file_name)};
    if (!input.is_open()) {
        throw std::invalid_argument("Cannot open file: " +
                                    std::string(file_name));
    }

    nlohmann::json root;
    input >> root;
    if (!root.is_object()) {
        throw std::invalid_argument(
            "Invalid JSON checkpoint: root must be an object");
    }

    model_layer_t model_state;
    for (auto it = root.begin(); it != root.end(); ++it) {
        if (!it.value().is_array() && !it.value().is_number()) {
            throw std::invalid_argument("Invalid tensor for key: " + it.key());
        }

        const auto dot_pos = it.key().rfind('.');
        if (dot_pos == std::string::npos || dot_pos == 0 ||
            dot_pos + 1 == it.key().size()) {
            throw std::invalid_argument(
                "Invalid tensor key format (expected layer.tensor): " +
                it.key());
        }

        const auto layer_name = it.key().substr(0, dot_pos);
        const auto tensor_name = it.key().substr(dot_pos + 1);

        Shape shape;
        std::vector<scalar_t> values;
        std::size_t leaf_depth = 0;
        bool has_leaf_depth = false;
        json2tensor(it.value(), shape, values, 0, leaf_depth, has_leaf_depth);

        model_state.layers[layer_name].tensors.emplace(
            tensor_name, tensor_t(shape, std::move(values)));
    }

    return model_state;
}

template <DType _DType, DeviceLikeType _Device>
void Loader<_DType, _Device, LoadLikeType::json>::json2tensor(
    const nlohmann::json& node, Shape& shape, std::vector<scalar_t>& data,
    std::size_t depth, std::size_t& leaf_depth, bool& has_leaf_depth) {
    if (node.is_array()) {
        if (shape.size() == depth) {
            shape.push_back(static_cast<std::int64_t>(node.size()));
        } else if (shape[depth] != static_cast<std::int64_t>(node.size())) {
            throw std::invalid_argument(
                "Invalid tensor JSON: ragged array is not supported");
        }

        for (const auto& child : node) {
            json2tensor(child, shape, data, depth + 1, leaf_depth,
                        has_leaf_depth);
        }
        return;
    }

    if (!node.is_number()) {
        throw std::invalid_argument(
            "Invalid tensor JSON: expected numeric value");
    }

    if (!has_leaf_depth) {
        leaf_depth = depth;
        has_leaf_depth = true;
    } else if (leaf_depth != depth) {
        throw std::invalid_argument(
            "Invalid tensor JSON: inconsistent nesting depth");
    }

    data.push_back(node.get<scalar_t>());
}
}  // namespace fastinf
