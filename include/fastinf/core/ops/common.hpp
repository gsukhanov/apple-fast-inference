#pragma once

#include "../device.hpp"
#include "../dtype.hpp"

namespace fastinf {
template <typename TLhs, typename TRhs>
void check_same_shape(const TLhs& lhs, const TRhs& rhs);
}  // namespace fastinf

#include "../../../../src/core/ops/common.hpp.inl"
