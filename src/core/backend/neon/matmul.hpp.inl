#include "fastinf/core/backend/common.hpp"
#include "fastinf/core/backend/neon/matmul.hpp"
#include "fastinf/core/backend/neon/quantize.hpp"
#include "fastinf/core/tensor.hpp"
#include "fastinf/core/backend/neon/blocking.hpp"

#include "arm_neon.h"

// goto labels
#define DOT_PRODUCT_LOOP "1"
#define DOT_PRODUCT_LOOP_END "2"

namespace fastinf {

template <DType _IntDType>
struct Mapper;

template <>
struct Mapper<DType::int8> {
    static constexpr DType accType = DType::int32;
    using vType = int8x16_t;
};

template <>
struct Mapper<DType::int16> {
    static constexpr DType accType = DType::int64;
    using vType = int16x8_t;
};

inline void KernelMxV(int32_t* dst_data, int8_t* lhs_data, int8_t* rhs_data) {
    int8x16_t rhs_register_1, rhs_register_2;
    int8x16_t rhs_register_3, rhs_register_4;
    int8x16_t rhs_register_5, rhs_register_6;
    int8x16_t rhs_register_7, rhs_register_8;

    int8x16_t lhs_register_1, lhs_register_2;
    int8x16_t lhs_register_3, lhs_register_4;
    int8x16_t lhs_register_5, lhs_register_6;
    int8x16_t lhs_register_7, lhs_register_8;
    int8x16_t lhs_register_9, lhs_register_10;
    int8x16_t lhs_register_11, lhs_register_12;
    int8x16_t lhs_register_13, lhs_register_14;
    int8x16_t lhs_register_15, lhs_register_16;


    int16x8_t dst_register_1, dst_register_2;
//        int16x8_t dst_register_3, dst_register_4;
//        int16x8_t dst_register_5, dst_register_6;
//        int16x8_t dst_register_7, dst_register_8;


    asm volatile (
        "ld1 {%[rhs_1].8b}, [%[rhs_ptr]]\n"
        "ld1 {%[rhs_2].8b}, [%[rhs_ptr], #8]\n"
        "ld1 {%[rhs_3].8b}, [%[rhs_ptr], #16]\n"
        "ld1 {%[rhs_4].8b}, [%[rhs_ptr], #24]\n"
        "ld1 {%[rhs_5].8b}, [%[rhs_ptr], #32]\n"
        "ld1 {%[rhs_6].8b}, [%[rhs_ptr], #40]\n"
        "ld1 {%[rhs_7].8b}, [%[rhs_ptr], #48]\n"
        "ld1 {%[rhs_8].8b}, [%[rhs_ptr], #56]\n"

        "ld1 {%[lhs_1].8b}, [%[lhs_ptr]]\n"
        "ld1 {%[lhs_2].8b}, [%[lhs_ptr], #8]\n"
        "ld1 {%[lhs_3].8b}, [%[lhs_ptr], #16]\n"
        "ld1 {%[lhs_4].8b}, [%[lhs_ptr], #24]\n"
        "ld1 {%[lhs_5].8b}, [%[lhs_ptr], #32]\n"
        "ld1 {%[lhs_6].8b}, [%[lhs_ptr], #40]\n"
        "ld1 {%[lhs_7].8b}, [%[lhs_ptr], #48]\n"
        "ld1 {%[lhs_8].8b}, [%[lhs_ptr], #56]\n"
        "ld1 {%[lhs_9].8b}, [%[lhs_ptr], #64]\n"
        "ld1 {%[lhs_10].8b}, [%[lhs_ptr], #72]\n"
        "ld1 {%[lhs_11].8b}, [%[lhs_ptr], #80]\n"
        "ld1 {%[lhs_12].8b}, [%[lhs_ptr], #88]\n"
        "ld1 {%[lhs_13].8b}, [%[lhs_ptr], #96]\n"
        "ld1 {%[lhs_14].8b}, [%[lhs_ptr], #104]\n"
        "ld1 {%[lhs_15].8b}, [%[lhs_ptr], #112]\n"
        "ld1 {%[lhs_16].8b}, [%[lhs_ptr], #120]\n"

        "smlal %[dst_1].4s, %[lhs_1].4b, %[rhs_1].4b\n"
        "smlal %[dst_1].4s, %[lhs_2].4b, %[rhs_2].4b\n"
        "smlal %[dst_1].4s, %[lhs_3].4b, %[rhs_3].4b\n"
        "smlal %[dst_1].4s, %[lhs_4].4b, %[rhs_4].4b\n"
        "smlal %[dst_1].4s, %[lhs_5].4b, %[rhs_5].4b\n"
        "smlal %[dst_1].4s, %[lhs_6].4b, %[rhs_6].4b\n"
        "smlal %[dst_1].4s, %[lhs_7].4b, %[rhs_7].4b\n"
        "smlal %[dst_1].4s, %[lhs_8].4b, %[rhs_8].4b\n"

        "smlal %[dst_2].4s, %[lhs_1].4b, %[rhs_1].4b\n"
        "smlal %[dst_2].4s, %[lhs_2].4b, %[rhs_2].4b\n"
        "smlal %[dst_2].4s, %[lhs_3].4b, %[rhs_3].4b\n"
        "smlal %[dst_2].4s, %[lhs_4].4b, %[rhs_4].4b\n"
        "smlal %[dst_2].4s, %[lhs_5].4b, %[rhs_5].4b\n"
        "smlal %[dst_2].4s, %[lhs_6].4b, %[rhs_6].4b\n"
        "smlal %[dst_2].4s, %[lhs_7].4b, %[rhs_7].4b\n"
        "smlal %[dst_2].4s, %[lhs_8].4b, %[rhs_8].4b\n"

        "st1 {%[dst_1].4s}, [%[dst_ptr]]\n"
        "st1 {%[dst_2].4s}, [%[dst_ptr], #32]\n"

        "smlal2 %[dst_1].4s, %[lhs_1].8b, %[rhs_1].8b\n"
        "smlal2 %[dst_1].4s, %[lhs_2].8b, %[rhs_2].8b\n"
        "smlal2 %[dst_1].4s, %[lhs_3].8b, %[rhs_3].8b\n"
        "smlal2 %[dst_1].4s, %[lhs_4].8b, %[rhs_4].8b\n"
        "smlal2 %[dst_1].4s, %[lhs_5].8b, %[rhs_5].8b\n"
        "smlal2 %[dst_1].4s, %[lhs_6].8b, %[rhs_6].8b\n"
        "smlal2 %[dst_1].4s, %[lhs_7].8b, %[rhs_7].8b\n"
        "smlal2 %[dst_1].4s, %[lhs_8].8b, %[rhs_8].8b\n"

        "smlal2 %[dst_2].4s, %[lhs_1].8b, %[rhs_1].8b\n"
        "smlal2 %[dst_2].4s, %[lhs_2].8b, %[rhs_2].8b\n"
        "smlal2 %[dst_2].4s, %[lhs_3].8b, %[rhs_3].8b\n"
        "smlal2 %[dst_2].4s, %[lhs_4].8b, %[rhs_4].8b\n"
        "smlal2 %[dst_2].4s, %[lhs_5].8b, %[rhs_5].8b\n"
        "smlal2 %[dst_2].4s, %[lhs_6].8b, %[rhs_6].8b\n"
        "smlal2 %[dst_2].4s, %[lhs_7].8b, %[rhs_7].8b\n"
        "smlal2 %[dst_2].4s, %[lhs_8].8b, %[rhs_8].8b\n"

        "st1 {%[dst_1].4s}, [%[dst_ptr], #16]\n"
        "st1 {%[dst_2].4s}, [%[dst_ptr], #48]\n"

        : [lhs_ptr] "+r" (lhs_data), [rhs_ptr] "+r" (rhs_data),

        [rhs_1] "=w" (rhs_register_1), [rhs_2] "=w" (rhs_register_2),
        [rhs_3] "=w" (rhs_register_3), [rhs_4] "=w" (rhs_register_4),
        [rhs_5] "=w" (rhs_register_5), [rhs_6] "=w" (rhs_register_6),
        [rhs_7] "=w" (rhs_register_7), [rhs_8] "=w" (rhs_register_8),

        [lhs_1] "=w" (lhs_register_1), [lhs_2] "=w" (lhs_register_2),
        [lhs_3] "=w" (lhs_register_3), [lhs_4] "=w" (lhs_register_4),
        [lhs_5] "=w" (lhs_register_5), [lhs_6] "=w" (lhs_register_6),
        [lhs_7] "=w" (lhs_register_7), [lhs_8] "=w" (lhs_register_8),
        [lhs_9] "=w" (lhs_register_9), [lhs_10] "=w" (lhs_register_10),
        [lhs_11] "=w" (lhs_register_11), [lhs_12] "=w" (lhs_register_12),
        [lhs_13] "=w" (lhs_register_13), [lhs_14] "=w" (lhs_register_14),
        [lhs_15] "=w" (lhs_register_15), [lhs_16] "=w" (lhs_register_16),

        [dst_1] "=w" (dst_register_1), [dst_2] "=w" (dst_register_2),

        [dst_ptr] "+r" (dst_data)
        :: "memory"
    );
}


inline void KernelVxM(int32_t* dst_data, int8_t* rhs_data, int8_t* lhs_data) {
    int8x16_t lhs_register_1, lhs_register_2;
    int8x16_t lhs_register_3, lhs_register_4;
    int8x16_t lhs_register_5, lhs_register_6;
    int8x16_t lhs_register_7, lhs_register_8;

    int8x16_t rhs_register_1, rhs_register_2;
    int8x16_t rhs_register_3, rhs_register_4;
    int8x16_t rhs_register_5, rhs_register_6;
    int8x16_t rhs_register_7, rhs_register_8;
    int8x16_t rhs_register_9, rhs_register_10;
    int8x16_t rhs_register_11, rhs_register_12;
    int8x16_t rhs_register_13, rhs_register_14;
    int8x16_t rhs_register_15, rhs_register_16;


    int16x8_t dst_register_1, dst_register_2;
//        int16x8_t dst_register_3, dst_register_4;
//        int16x8_t dst_register_5, dst_register_6;
//        int16x8_t dst_register_7, dst_register_8;


    asm volatile (
        "ld1 {%[lhs_1].8b}, [%[lhs_ptr]]\n"
        "ld1 {%[lhs_2].8b}, [%[lhs_ptr], #8]\n"
        "ld1 {%[lhs_3].8b}, [%[lhs_ptr], #16]\n"
        "ld1 {%[lhs_4].8b}, [%[lhs_ptr], #24]\n"
        "ld1 {%[lhs_5].8b}, [%[lhs_ptr], #32]\n"
        "ld1 {%[lhs_6].8b}, [%[lhs_ptr], #40]\n"
        "ld1 {%[lhs_7].8b}, [%[lhs_ptr], #48]\n"
        "ld1 {%[lhs_6].8b}, [%[lhs_ptr], #56]\n"

        "ld1 {%[rhs_1].8b}, [%[rhs_ptr]]\n"
        "ld1 {%[rhs_2].8b}, [%[rhs_ptr], #8]\n"
        "ld1 {%[rhs_3].8b}, [%[rhs_ptr], #16]\n"
        "ld1 {%[rhs_4].8b}, [%[rhs_ptr], #24]\n"
        "ld1 {%[rhs_5].8b}, [%[rhs_ptr], #32]\n"
        "ld1 {%[rhs_6].8b}, [%[rhs_ptr], #40]\n"
        "ld1 {%[rhs_7].8b}, [%[rhs_ptr], #48]\n"
        "ld1 {%[rhs_8].8b}, [%[rhs_ptr], #56]\n"
        "ld1 {%[rhs_9].8b}, [%[rhs_ptr], #64]\n"
        "ld1 {%[rhs_10].8b}, [%[rhs_ptr], #72]\n"
        "ld1 {%[rhs_11].8b}, [%[rhs_ptr], #80]\n"
        "ld1 {%[rhs_12].8b}, [%[rhs_ptr], #88]\n"
        "ld1 {%[rhs_13].8b}, [%[rhs_ptr], #96]\n"
        "ld1 {%[rhs_14].8b}, [%[rhs_ptr], #104]\n"
        "ld1 {%[rhs_15].8b}, [%[rhs_ptr], #112]\n"
        "ld1 {%[rhs_16].8b}, [%[rhs_ptr], #120]\n"

        "ld1 {%[dst_1].4s}, [%[dst_ptr]]\n"
        "ld1 {%[dst_2].4s}, [%[dst_ptr], #16]\n"

        "smlal %[dst_1].4s, %[rhs_1].4b, %[lhs_1].4b\n"
        "smlal %[dst_1].4s, %[rhs_2].4b, %[lhs_2].4b\n"
        "smlal %[dst_1].4s, %[rhs_3].4b, %[lhs_3].4b\n"
        "smlal %[dst_1].4s, %[rhs_4].4b, %[lhs_4].4b\n"
        "smlal %[dst_1].4s, %[rhs_5].4b, %[lhs_5].4b\n"
        "smlal %[dst_1].4s, %[rhs_6].4b, %[lhs_6].4b\n"
        "smlal %[dst_1].4s, %[rhs_7].4b, %[lhs_7].4b\n"
        "smlal %[dst_1].4s, %[rhs_8].4b, %[lhs_8].4b\n"

        "smlal %[dst_2].4s, %[rhs_1].4b, %[lhs_1].4b\n"
        "smlal %[dst_2].4s, %[rhs_2].4b, %[lhs_2].4b\n"
        "smlal %[dst_2].4s, %[rhs_3].4b, %[lhs_3].4b\n"
        "smlal %[dst_2].4s, %[rhs_4].4b, %[lhs_4].4b\n"
        "smlal %[dst_2].4s, %[rhs_5].4b, %[lhs_5].4b\n"
        "smlal %[dst_2].4s, %[rhs_6].4b, %[lhs_6].4b\n"
        "smlal %[dst_2].4s, %[rhs_7].4b, %[lhs_7].4b\n"
        "smlal %[dst_2].4s, %[rhs_8].4b, %[lhs_8].4b\n"

        "smlal2 %[dst_1].4s, %[rhs_1].8b, %[lhs_1].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_2].8b, %[lhs_2].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_3].8b, %[lhs_3].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_4].8b, %[lhs_4].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_5].8b, %[lhs_5].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_6].8b, %[lhs_6].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_7].8b, %[lhs_7].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_8].8b, %[lhs_8].8b\n"

        "smlal2 %[dst_2].4s, %[rhs_1].8b, %[lhs_1].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_2].8b, %[lhs_2].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_3].8b, %[lhs_3].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_4].8b, %[lhs_4].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_5].8b, %[lhs_5].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_6].8b, %[lhs_6].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_7].8b, %[lhs_7].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_8].8b, %[lhs_8].8b\n"

        "st1 {%[dst_1].4s}, [%[dst_ptr]]\n"
        "st1 {%[dst_2].4s}, [%[dst_ptr], #16]\n"

        : [lhs_ptr] "+r" (lhs_data), [rhs_ptr] "+r" (rhs_data), [dst_ptr] "+r" (dst_data),

        [lhs_1] "=w" (lhs_register_1), [lhs_2] "=w" (lhs_register_2),
        [lhs_3] "=w" (lhs_register_3), [lhs_4] "=w" (lhs_register_4),
        [lhs_5] "=w" (lhs_register_5), [lhs_6] "=w" (lhs_register_6),
        [lhs_7] "=w" (lhs_register_7), [lhs_8] "=w" (lhs_register_8),

        [rhs_1] "=w" (rhs_register_1), [rhs_2] "=w" (rhs_register_2),
        [rhs_3] "=w" (rhs_register_3), [rhs_4] "=w" (rhs_register_4),
        [rhs_5] "=w" (rhs_register_5), [rhs_6] "=w" (rhs_register_6),
        [rhs_7] "=w" (rhs_register_7), [rhs_8] "=w" (rhs_register_8),
        [rhs_9] "=w" (rhs_register_9), [rhs_10] "=w" (rhs_register_10),
        [rhs_11] "=w" (rhs_register_11), [rhs_12] "=w" (rhs_register_12),
        [rhs_13] "=w" (rhs_register_13), [rhs_14] "=w" (rhs_register_14),
        [rhs_15] "=w" (rhs_register_15), [rhs_16] "=w" (rhs_register_16),

        [dst_1] "=w" (dst_register_1), [dst_2] "=w" (dst_register_2)
        :: "memory"
    );
}



inline void KernelMxM(int32_t* dst_data, int8_t* lhs_data, int8_t* rhs_data) {
    int8x16_t lhs_register_1, lhs_register_2;
    int8x16_t lhs_register_3, lhs_register_4;
    int8x16_t lhs_register_5, lhs_register_6;
    int8x16_t lhs_register_7, lhs_register_8;
    int8x16_t lhs_register_9, lhs_register_10;
    int8x16_t lhs_register_11, lhs_register_12;

    int8x16_t rhs_register_1, rhs_register_2;
    int8x16_t rhs_register_3, rhs_register_4;
    int8x16_t rhs_register_5, rhs_register_6;
    int8x16_t rhs_register_7, rhs_register_8;
    int8x16_t rhs_register_9, rhs_register_10;
    int8x16_t rhs_register_11, rhs_register_12;

    int16x8_t dst_register_1, dst_register_2, dst_register_3;
    int16x8_t dst_register_4, dst_register_5, dst_register_6;
    int16x8_t dst_register_7, dst_register_8, dst_register_9;

        asm volatile (
        "ld1 {%[lhs_1].8b}, [%[lhs_ptr]]\n"
        "ld1 {%[lhs_2].8b}, [%[lhs_ptr], #8]\n"
        "ld1 {%[lhs_3].8b}, [%[lhs_ptr], #16]\n"
        "ld1 {%[lhs_4].8b}, [%[lhs_ptr], #24]\n"
        "ld1 {%[lhs_5].8b}, [%[lhs_ptr], #32]\n"
        "ld1 {%[lhs_6].8b}, [%[lhs_ptr], #40]\n"
        "ld1 {%[lhs_7].8b}, [%[lhs_ptr], #48]\n"
        "ld1 {%[lhs_8].8b}, [%[lhs_ptr], #56]\n"
        "ld1 {%[lhs_9].8b}, [%[lhs_ptr], #64]\n"
        "ld1 {%[lhs_10].8b}, [%[lhs_ptr], #72]\n"
        "ld1 {%[lhs_11].8b}, [%[lhs_ptr], #80]\n"
        "ld1 {%[lhs_12].8b}, [%[lhs_ptr], #88]\n"

        "ld1 {%[rhs_1].8b}, [%[rhs_ptr]]\n"
        "ld1 {%[rhs_2].8b}, [%[rhs_ptr], #8]\n"
        "ld1 {%[rhs_3].8b}, [%[rhs_ptr], #16]\n"
        "ld1 {%[rhs_4].8b}, [%[rhs_ptr], #24]\n"
        "ld1 {%[rhs_5].8b}, [%[rhs_ptr], #32]\n"
        "ld1 {%[rhs_6].8b}, [%[rhs_ptr], #40]\n"
        "ld1 {%[rhs_7].8b}, [%[rhs_ptr], #48]\n"
        "ld1 {%[rhs_8].8b}, [%[rhs_ptr], #56]\n"
        "ld1 {%[rhs_9].8b}, [%[rhs_ptr], #64]\n"
        "ld1 {%[rhs_10].8b}, [%[rhs_ptr], #72]\n"
        "ld1 {%[rhs_11].8b}, [%[rhs_ptr], #80]\n"
        "ld1 {%[rhs_12].8b}, [%[rhs_ptr], #88]\n"

        "ld1 {%[dst_1].4s}, [%[dst_ptr]]\n"
        "ld1 {%[dst_2].4s}, [%[dst_ptr], #16]\n"
        "ld1 {%[dst_3].4s}, [%[dst_ptr], #32]\n"
        "ld1 {%[dst_4].4s}, [%[dst_ptr], #48]\n"
        "ld1 {%[dst_5].4s}, [%[dst_ptr], #64]\n"
        "ld1 {%[dst_6].4s}, [%[dst_ptr], #80]\n"
        "ld1 {%[dst_7].4s}, [%[dst_ptr], #96]\n"
        "ld1 {%[dst_8].4s}, [%[dst_ptr], #112]\n"
        "ld1 {%[dst_9].4s}, [%[dst_ptr], #128]\n"

        "smlal %[dst_1].4s, %[rhs_1].4b, %[lhs_1].4b\n"
        "smlal %[dst_1].4s, %[rhs_2].4b, %[lhs_2].4b\n"
        "smlal %[dst_1].4s, %[rhs_3].4b, %[lhs_3].4b\n"
        "smlal %[dst_1].4s, %[rhs_4].4b, %[lhs_4].4b\n"

        "smlal2 %[dst_1].4s, %[rhs_1].8b, %[lhs_1].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_2].8b, %[lhs_2].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_3].8b, %[lhs_3].8b\n"
        "smlal2 %[dst_1].4s, %[rhs_4].8b, %[lhs_4].8b\n"

        "smlal %[dst_2].4s, %[rhs_5].4b, %[lhs_1].4b\n"
        "smlal %[dst_2].4s, %[rhs_6].4b, %[lhs_2].4b\n"
        "smlal %[dst_2].4s, %[rhs_7].4b, %[lhs_3].4b\n"
        "smlal %[dst_2].4s, %[rhs_8].4b, %[lhs_4].4b\n"

        "smlal2 %[dst_2].4s, %[rhs_5].8b, %[lhs_1].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_6].8b, %[lhs_2].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_7].8b, %[lhs_3].8b\n"
        "smlal2 %[dst_2].4s, %[rhs_8].8b, %[lhs_4].8b\n"

        "smlal %[dst_3].4s, %[rhs_9].4b, %[lhs_1].4b\n"
        "smlal %[dst_3].4s, %[rhs_10].4b, %[lhs_2].4b\n"
        "smlal %[dst_3].4s, %[rhs_11].4b, %[lhs_3].4b\n"
        "smlal %[dst_3].4s, %[rhs_12].4b, %[lhs_4].4b\n"

        "smlal2 %[dst_3].4s, %[rhs_9].8b, %[lhs_1].8b\n"
        "smlal2 %[dst_3].4s, %[rhs_10].8b, %[lhs_2].8b\n"
        "smlal2 %[dst_3].4s, %[rhs_11].8b, %[lhs_3].8b\n"
        "smlal2 %[dst_3].4s, %[rhs_12].8b, %[lhs_4].8b\n"

        "smlal %[dst_4].4s, %[rhs_1].4b, %[lhs_5].4b\n"
        "smlal %[dst_4].4s, %[rhs_2].4b, %[lhs_6].4b\n"
        "smlal %[dst_4].4s, %[rhs_3].4b, %[lhs_7].4b\n"
        "smlal %[dst_4].4s, %[rhs_4].4b, %[lhs_8].4b\n"

        "smlal2 %[dst_4].4s, %[rhs_1].8b, %[lhs_5].8b\n"
        "smlal2 %[dst_4].4s, %[rhs_2].8b, %[lhs_6].8b\n"
        "smlal2 %[dst_4].4s, %[rhs_3].8b, %[lhs_7].8b\n"
        "smlal2 %[dst_4].4s, %[rhs_4].8b, %[lhs_8].8b\n"

        "smlal %[dst_5].4s, %[rhs_5].4b, %[lhs_5].4b\n"
        "smlal %[dst_5].4s, %[rhs_6].4b, %[lhs_6].4b\n"
        "smlal %[dst_5].4s, %[rhs_7].4b, %[lhs_7].4b\n"
        "smlal %[dst_5].4s, %[rhs_8].4b, %[lhs_8].4b\n"

        "smlal2 %[dst_5].4s, %[rhs_5].8b, %[lhs_5].8b\n"
        "smlal2 %[dst_5].4s, %[rhs_6].8b, %[lhs_6].8b\n"
        "smlal2 %[dst_5].4s, %[rhs_7].8b, %[lhs_7].8b\n"
        "smlal2 %[dst_5].4s, %[rhs_8].8b, %[lhs_8].8b\n"

        "smlal %[dst_6].4s, %[rhs_9].4b, %[lhs_5].4b\n"
        "smlal %[dst_6].4s, %[rhs_10].4b, %[lhs_6].4b\n"
        "smlal %[dst_6].4s, %[rhs_11].4b, %[lhs_7].4b\n"
        "smlal %[dst_6].4s, %[rhs_12].4b, %[lhs_8].4b\n"

        "smlal2 %[dst_6].4s, %[rhs_9].8b, %[lhs_5].8b\n"
        "smlal2 %[dst_6].4s, %[rhs_10].8b, %[lhs_6].8b\n"
        "smlal2 %[dst_6].4s, %[rhs_11].8b, %[lhs_7].8b\n"
        "smlal2 %[dst_6].4s, %[rhs_12].8b, %[lhs_8].8b\n"


        "smlal %[dst_7].4s, %[rhs_1].4b, %[lhs_9].4b\n"
        "smlal %[dst_7].4s, %[rhs_2].4b, %[lhs_10].4b\n"
        "smlal %[dst_7].4s, %[rhs_3].4b, %[lhs_11].4b\n"
        "smlal %[dst_7].4s, %[rhs_4].4b, %[lhs_12].4b\n"

        "smlal2 %[dst_7].4s, %[rhs_1].8b, %[lhs_9].8b\n"
        "smlal2 %[dst_7].4s, %[rhs_2].8b, %[lhs_10].8b\n"
        "smlal2 %[dst_7].4s, %[rhs_3].8b, %[lhs_11].8b\n"
        "smlal2 %[dst_7].4s, %[rhs_4].8b, %[lhs_12].8b\n"

        "smlal %[dst_8].4s, %[rhs_5].4b, %[lhs_9].4b\n"
        "smlal %[dst_8].4s, %[rhs_6].4b, %[lhs_10].4b\n"
        "smlal %[dst_8].4s, %[rhs_7].4b, %[lhs_11].4b\n"
        "smlal %[dst_8].4s, %[rhs_8].4b, %[lhs_12].4b\n"

        "smlal2 %[dst_8].4s, %[rhs_5].8b, %[lhs_9].8b\n"
        "smlal2 %[dst_8].4s, %[rhs_6].8b, %[lhs_10].8b\n"
        "smlal2 %[dst_8].4s, %[rhs_7].8b, %[lhs_11].8b\n"
        "smlal2 %[dst_8].4s, %[rhs_8].8b, %[lhs_12].8b\n"

        "smlal %[dst_9].4s, %[rhs_9].4b, %[lhs_9].4b\n"
        "smlal %[dst_9].4s, %[rhs_10].4b, %[lhs_10].4b\n"
        "smlal %[dst_9].4s, %[rhs_11].4b, %[lhs_11].4b\n"
        "smlal %[dst_9].4s, %[rhs_12].4b, %[lhs_12].4b\n"

        "smlal2 %[dst_9].4s, %[rhs_9].8b, %[lhs_9].8b\n"
        "smlal2 %[dst_9].4s, %[rhs_10].8b, %[lhs_10].8b\n"
        "smlal2 %[dst_9].4s, %[rhs_11].8b, %[lhs_11].8b\n"
        "smlal2 %[dst_9].4s, %[rhs_12].8b, %[lhs_12].8b\n"

        "st1 {%[dst_1].4s}, [%[dst_ptr]]\n"
        "st1 {%[dst_2].4s}, [%[dst_ptr], #16]\n"
        "st1 {%[dst_3].4s}, [%[dst_ptr], #32]\n"
        "st1 {%[dst_4].4s}, [%[dst_ptr], #48]\n"
        "st1 {%[dst_5].4s}, [%[dst_ptr], #64]\n"
        "st1 {%[dst_6].4s}, [%[dst_ptr], #80]\n"
        "st1 {%[dst_7].4s}, [%[dst_ptr], #96]\n"
        "st1 {%[dst_8].4s}, [%[dst_ptr], #112]\n"
        "st1 {%[dst_9].4s}, [%[dst_ptr], #128]\n"

        : [lhs_ptr] "+r" (lhs_data), [rhs_ptr] "+r" (rhs_data),

        [lhs_1] "=w" (lhs_register_1), [lhs_2] "=w" (lhs_register_2),
        [lhs_3] "=w" (lhs_register_3), [lhs_4] "=w" (lhs_register_4),
        [lhs_5] "=w" (lhs_register_5), [lhs_6] "=w" (lhs_register_6),
        [lhs_7] "=w" (lhs_register_7), [lhs_8] "=w" (lhs_register_8),
        [lhs_9] "=w" (lhs_register_9), [lhs_10] "=w" (lhs_register_10),
        [lhs_11] "=w" (lhs_register_11), [lhs_12] "=w" (lhs_register_12),

        [rhs_1] "=w" (rhs_register_1), [rhs_2] "=w" (rhs_register_2),
        [rhs_3] "=w" (rhs_register_3), [rhs_4] "=w" (rhs_register_4),
        [rhs_5] "=w" (rhs_register_5), [rhs_6] "=w" (rhs_register_6),
        [rhs_7] "=w" (rhs_register_7), [rhs_8] "=w" (rhs_register_8),
        [rhs_9] "=w" (rhs_register_9), [rhs_10] "=w" (rhs_register_10),
        [rhs_11] "=w" (rhs_register_11), [rhs_12] "=w" (rhs_register_12),

        [dst_1] "=w" (dst_register_1), [dst_2] "=w" (dst_register_2),
        [dst_3] "=w" (dst_register_3), [dst_4] "=w" (dst_register_4),
        [dst_5] "=w" (dst_register_5), [dst_6] "=w" (dst_register_6),
        [dst_7] "=w" (dst_register_7), [dst_8] "=w" (dst_register_8),
        [dst_9] "=w" (dst_register_9),

        [dst_ptr] "+r" (dst_data)
        :: "memory"
    );
}


template <DType _IntDType, DType _FloatDType>
Tensor<_IntDType, DeviceLikeType::neon> matmul(
    const TensorView<_IntDType, DeviceLikeType::neon>& lhs,
    const TensorView<_IntDType, DeviceLikeType::neon>& rhs,
    const QuantizeDesc<_IntDType, _FloatDType> lhs_qdesc,
    const QuantizeDesc<_IntDType, _FloatDType> rhs_qdesc) {
    using tensor_t = Tensor<_IntDType, DeviceLikeType::neon>;
    using scalar_t = typename DTypeTraits<_IntDType>::type;
    using accum_t = typename DTypeTraits<Mapper<_IntDType>::accType>::type;
    using accum_tensor_t =
        Tensor<Mapper<_IntDType>::accType, DeviceLikeType::neon>;
    using float_t = typename DTypeTraits<_FloatDType>::type;

    tensor_t lhs_tmp, rhs_tmp;
    scalar_t* lhs_data = backend::ensure_contiguous_data(lhs, lhs_tmp);
    scalar_t* rhs_data = backend::ensure_contiguous_data(rhs, rhs_tmp);
    (void)lhs_data;
    (void)rhs_data;

    if (lhs.dim() > 2 || rhs.dim() > 2) {
        throw std::runtime_error(
            "The size of tensor a and tensor b must be less than 3");
    }

    const bool lhs_is_vector = lhs.dim() == 1;
    const bool rhs_is_vector = rhs.dim() == 1;
    const std::int64_t m = lhs_is_vector ? 1 : lhs.shape()[0];
    const std::int64_t k_lhs = lhs_is_vector ? lhs.shape()[0] : lhs.shape()[1];
    const std::int64_t k_rhs = rhs_is_vector ? rhs.shape()[0] : rhs.shape()[0];
    const std::int64_t n = rhs_is_vector ? 1 : rhs.shape()[1];

    if (k_lhs != k_rhs) {
        throw std::runtime_error(
            "The size of tensor a must match the size of tensor b "
            "at non-singleton dimension");
    }

    Shape result_shape;
    if (lhs_is_vector && rhs_is_vector) {
        result_shape = Shape{};
    } else if (rhs_is_vector) {
        result_shape = Shape{m};
    } else if (lhs_is_vector) {
        result_shape = Shape{n};
    } else {
        result_shape = Shape{m, n};
    }

    accum_tensor_t accum_matrix(Shape{m, n}, accum_t{0});

    for (std::int64_t l2_row = 0; l2_row < m; l2_row += L2_BLOCK_WIDTH) {
        const std::int64_t l2_row_end =
            std::min<std::int64_t>(l2_row + L2_BLOCK_WIDTH, m);
        for (std::int64_t l2_col = 0; l2_col < n; l2_col += L2_BLOCK_WIDTH) {
            const std::int64_t l2_col_end =
                std::min<std::int64_t>(l2_col + L2_BLOCK_WIDTH, n);
            for (std::int64_t l2_depth = 0; l2_depth < k_lhs;
                 l2_depth += L2_BLOCK_DEPTH) {
                const std::int64_t l2_depth_end =
                    std::min<std::int64_t>(l2_depth + L2_BLOCK_DEPTH, k_lhs);

                for (std::int64_t l1_row = l2_row; l1_row < l2_row_end;
                     l1_row += L1_BLOCK_WIDTH) {
                    const std::int64_t l1_row_end =
                        std::min<std::int64_t>(l1_row + L1_BLOCK_WIDTH,
                                               l2_row_end);
                    for (std::int64_t l1_col = l2_col; l1_col < l2_col_end;
                         l1_col += L1_BLOCK_WIDTH) {
                        const std::int64_t l1_col_end =
                            std::min<std::int64_t>(l1_col + L1_BLOCK_WIDTH,
                                                   l2_col_end);
                        for (std::int64_t l1_depth = l2_depth;
                             l1_depth < l2_depth_end;
                             l1_depth += L1_BLOCK_DEPTH) {
                            const std::int64_t l1_depth_end =
                                std::min<std::int64_t>(
                                    l1_depth + L1_BLOCK_DEPTH, l2_depth_end);

                            for (std::int64_t row = l1_row; row < l1_row_end;
                                 row += KERNEL_RESULT_WIDTH) {
                                const std::int64_t row_end =
                                    std::min<std::int64_t>(
                                        row + KERNEL_RESULT_WIDTH, l1_row_end);
                                for (std::int64_t col = l1_col;
                                     col < l1_col_end;
                                     col += KERNEL_RESULT_WIDTH) {
                                    const std::int64_t col_end =
                                        std::min<std::int64_t>(
                                            col + KERNEL_RESULT_WIDTH,
                                            l1_col_end);
                                    for (std::int64_t depth = l1_depth;
                                         depth < l1_depth_end;
                                         depth += KERNEL_DEPTH) {
                                        const std::int64_t depth_end =
                                            std::min<std::int64_t>(
                                                depth + KERNEL_DEPTH,
                                                l1_depth_end);

                                        for (std::int64_t i = row;
                                             i < row_end; ++i) {
                                            for (std::int64_t j = col;
                                                 j < col_end; ++j) {
                                                accum_t sum = accum_t{0};
                                                for (std::int64_t kk = depth;
                                                     kk < depth_end; ++kk) {
                                                    const scalar_t lhs_value =
                                                        lhs_is_vector
                                                            ? lhs.at(Shape{kk})
                                                            : lhs.at(
                                                                  Shape{i, kk});
                                                    const scalar_t rhs_value =
                                                        rhs_is_vector
                                                            ? rhs.at(Shape{kk})
                                                            : rhs.at(
                                                                  Shape{kk, j});
                                                    sum += static_cast<accum_t>(
                                                               lhs_value -
                                                               lhs_qdesc
                                                                   .zero_point) *
                                                           static_cast<accum_t>(
                                                               rhs_value -
                                                               rhs_qdesc
                                                                   .zero_point);
                                                }
                                                accum_matrix.at(Shape{i, j}) +=
                                                    sum;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    auto packed_accum = pack_result(accum_matrix);
    auto unpacked_accum = unpack_result(packed_accum, Shape{m, n});

    const float_t output_scale =
        lhs_qdesc.scale * rhs_qdesc.scale *
        static_cast<float_t>(DTypeTraits<_IntDType>::max);
    auto accum_pair = std::make_pair(
        unpacked_accum,
        QuantizeDesc<_IntDType, _FloatDType>(0, output_scale));
    auto scaled_pair =
        rescale<Mapper<_IntDType>::accType, _IntDType, _FloatDType>(
            accum_pair);
    tensor_t scaled_matrix = std::move(scaled_pair.first);

    if (result_shape.empty()) {
        tensor_t result(result_shape, scalar_t{0});
        result.at(Shape{}) = scaled_matrix.at(Shape{0, 0});
        if (scaled_matrix.quantization().has_value()) {
            result.set_quantization(*scaled_matrix.quantization());
        }
        return result;
    }

    if (result_shape.size() == 1) {
        tensor_t result(result_shape, scalar_t{0});
        if (rhs_is_vector) {
            for (std::int64_t i = 0; i < m; ++i) {
                result.at(Shape{i}) = scaled_matrix.at(Shape{i, 0});
            }
        } else {
            for (std::int64_t j = 0; j < n; ++j) {
                result.at(Shape{j}) = scaled_matrix.at(Shape{0, j});
            }
        }
        if (scaled_matrix.quantization().has_value()) {
            result.set_quantization(*scaled_matrix.quantization());
        }
        return result;
    }

    return scaled_matrix;
}
}  // namespace fastinf
