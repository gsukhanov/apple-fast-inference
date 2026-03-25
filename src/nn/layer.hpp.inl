#include "fastinf/nn/layer.hpp"

namespace fastinf {
namespace nn {
template <DType _DType, DeviceLikeType _Device>
void Layer<_DType, _Device>::print(std::ostream& o, int level) const {
    (void)level;
    o << name();
}

template <DType _DType, DeviceLikeType _Device>
std::ostream& operator<<(std::ostream& o, const Layer<_DType, _Device>& m) {
    m.print(o);
    o << std::endl;
    return o;
}
}  // namespace nn
}  // namespace fastinf
