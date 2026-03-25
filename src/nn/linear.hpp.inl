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
typename Linear<_DType, _Device>::tensor_t Linear<_DType, _Device>::forward(
    const tensor_t& input) const {
    if (input.dim() != 2) {
        throw std::invalid_argument("Linear expects 2D input [N, in_features]");
    }

    const auto& shape = input.shape();
    const auto batch = shape[0];
    const auto in_features = shape[1];

    if (in_features != in_features_) {
        throw std::invalid_argument("Input features do not match Linear layer");
    }

    tensor_t output({batch, out_features_}, scalar_t{});

    for (std::int64_t n = 0; n < batch; ++n) {
        for (std::int64_t o = 0; o < out_features_; ++o) {
            scalar_t acc{};
            for (std::int64_t i = 0; i < in_features_; ++i) {
                acc += input.at({n, i}) * weight_.at({o, i});
            }
            if (use_bias_) {
                acc += bias_.at({o});
            }
            output.at({n, o}) = acc;
        }
    }

    return output;
}

template <DType _DType, DeviceLikeType _Device>
const typename Linear<_DType, _Device>::tensor_t&
Linear<_DType, _Device>::weight()
    const {
    return weight_;
}

template <DType _DType, DeviceLikeType _Device>
const typename Linear<_DType, _Device>::tensor_t&
Linear<_DType, _Device>::bias() const {
    return bias_;
}

template <DType _DType, DeviceLikeType _Device>
void Linear<_DType, _Device>::load_weights(const tensor_t& weight,
                                           const tensor_t& bias) {
    if (weight.shape() != Shape{out_features_, in_features_}) {
        throw std::invalid_argument("Linear weight shape mismatch");
    }
    if (bias.shape() != Shape{out_features_}) {
        throw std::invalid_argument("Linear bias shape mismatch");
    }
    weight_ = weight;
    bias_ = bias;
}

}  // namespace nn
}  // namespace fastinf
