#pragma once

#include <cstdint>
#include <type_traits>

namespace fastinf {
enum DType { int8, int16, int32, int64, float32, float64 };

template <DType>
struct DTypeTraits;

template <>
struct DTypeTraits<DType::int8> {
    using type = std::int8_t;
    static constexpr std::size_t size = 1;
    static constexpr type max = __INT8_MAX__; 
};
template <>
struct DTypeTraits<DType::int16> {
    using type = std::int16_t;
    static constexpr std::size_t size = 2;
    static constexpr type max = __INT16_MAX__; 
};
template <>
struct DTypeTraits<DType::int32> {
    using type = std::int32_t;
    static constexpr std::size_t size = 4;
    static constexpr type max = __INT32_MAX__; 
};
template <>
struct DTypeTraits<DType::int64> {
    using type = std::int64_t;
    static constexpr std::size_t size = 8;
    static constexpr type max = __INT64_MAX__; 
};
template <>
struct DTypeTraits<DType::float32> {
    using type = float;
    static constexpr std::size_t size = 4;
    static constexpr type max = __FLT_MAX__;
};
template <>
struct DTypeTraits<DType::float64> {
    using type = double;
    static constexpr std::size_t size = 8;
    static constexpr type max = __DBL_MAX__;
};

};  // namespace fastinf
