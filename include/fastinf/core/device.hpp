#pragma once

#include <cstdint>

namespace fastinf {
enum class DeviceLikeType : std::uint8_t { cpu, neon, amx };

template <DeviceLikeType>
struct Device {};
using cpu_t = Device<DeviceLikeType::cpu>;
using neon_t = Device<DeviceLikeType::neon>;
using amx_t = Device<DeviceLikeType::amx>;

};  // namespace fastinf
