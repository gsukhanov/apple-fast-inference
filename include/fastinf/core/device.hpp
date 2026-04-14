#pragma once

#include <cstdint>

namespace fastinf {
enum class DeviceLikeType : std::uint8_t { cpu, greedy, neon, amx };

template <DeviceLikeType>
struct Device {};
using cpu_t = Device<DeviceLikeType::cpu>;
using greedy_t = Device<DeviceLikeType::cpu>;
using neon_t = Device<DeviceLikeType::neon>;
using amx_t = Device<DeviceLikeType::amx>;

};  // namespace fastinf
