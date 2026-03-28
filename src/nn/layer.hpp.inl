#include <utility>

#include "fastinf/nn/layer.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
typename Layer<_DType, _Device>::tensor_t Layer<_DType, _Device>::operator()(
    const tensor_t& input) const {
    return forward(input);
}

template <DType _DType, DeviceLikeType _Device>
void Layer<_DType, _Device>::print(std::ostream& o, int level) const {
    (void)level;
    if (key_.empty()) {
        o << name();
    } else {
        o << key_;
    }
}

template <DType _DType, DeviceLikeType _Device>
bool Layer<_DType, _Device>::load_state(const layer_state_t& state) {
    (void)state;
    return false;
}

template <DType _DType, DeviceLikeType _Device>
bool Layer<_DType, _Device>::load_state(const model_layer_t& state) {
    bool has_target = false;
    bool ok = true;

    if (!key_.empty() && state.contains(key_)) {
        has_target = true;
        ok = load_state(state.find(key_)) && ok;
    }

    for (auto* child : children_) {
        has_target = true;
        ok = child->load_state(state) && ok;
    }

    return has_target && ok;
}

template <DType _DType, DeviceLikeType _Device>
template <typename... _Layers>
void Layer<_DType, _Device>::bind_children(_Layers&... layers) {
    (children_.push_back(static_cast<Layer<_DType, _Device>*>(&layers)), ...);
}

template <DType _DType, DeviceLikeType _Device>
void Layer<_DType, _Device>::bind_identity(std::string key) {
    key_ = std::move(key);
}

template <DType _DType, DeviceLikeType _Device>
const std::string& Layer<_DType, _Device>::key() const {
    return key_;
}

template <DType _DType, DeviceLikeType _Device>
std::ostream& operator<<(std::ostream& o, const Layer<_DType, _Device>& m) {
    m.print(o);
    o << std::endl;
    return o;
}
}  // namespace nn
}  // namespace fastinf
