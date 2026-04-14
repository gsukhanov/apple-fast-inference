#include "fastinf/nn/linear.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
Linear<_DType, _Device>::Linear(std::int64_t in_features,
                                std::int64_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument(
            "Linear in_features and out_features must be positive");
    }

    weight_ = tensor_t({out_features, in_features});
    bias_ = tensor_t({out_features}, scalar_t{});

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    for (auto& el : weight_) {
        el = static_cast<scalar_t>(dist(gen));
    }

    if (use_bias_) {
        for (auto& el : bias_) {
            el = static_cast<scalar_t>(dist(gen));
        }
    }
}

template <DType _DType, DeviceLikeType _Device>
std::string Linear<_DType, _Device>::name() const {
    return "nn.Linear";
}

template <DType _DType, DeviceLikeType _Device>
bool Linear<_DType, _Device>::load_state(const layer_state_t& state) {
    if (!state.contains("weight")) {
        return false;
    }

    tensor_t weight = state.find("weight");
    tensor_t bias({out_features_}, scalar_t{});
    if (use_bias_) {
        if (!state.contains("bias")) {
            throw std::invalid_argument("Linear layer expects bias tensor");
        }
        bias = std::move(state.find("bias"));
    }

    if (weight.shape() != Shape{out_features_, in_features_}) {
        throw std::invalid_argument("Linear weight shape mismatch");
    }
    if (bias.shape() != Shape{out_features_}) {
        throw std::invalid_argument("Linear bias shape mismatch");
    }
    weight_ = std::move(weight);
    bias_ = std::move(bias);
    return true;
}

template <DType _DType, DeviceLikeType _Device>
typename Linear<_DType, _Device>::tensor_t Linear<_DType, _Device>::forward(
    const tensor_t& input) const {
    if (weight_.shape() != Shape{out_features_, in_features_}) {
        throw std::invalid_argument("Linear weight shape mismatch");
    }
    if (use_bias_ && bias_.shape() != Shape{out_features_}) {
        throw std::invalid_argument("Linear bias shape mismatch");
    }

    const auto input_view = input.view();
    const auto weight_view = weight_.view();
    if (!use_bias_) {
        return nn::linear<_DType, _Device>(input_view, weight_view);
    }

    const auto bias_view = bias_.view();
    return nn::linear<_DType, _Device>(input_view, weight_view,
                                       std::cref(bias_view));
}

template <DType _DType, DeviceLikeType _Device>
const typename Linear<_DType, _Device>::tensor_t&
Linear<_DType, _Device>::weight() const {
    return weight_;
}

template <DType _DType, DeviceLikeType _Device>
const typename Linear<_DType, _Device>::tensor_t&
Linear<_DType, _Device>::bias() const {
    return bias_;
}

}  // namespace nn
}  // namespace fastinf
