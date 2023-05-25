/*
 * Copyright (c) 2022, Gregory Bertilson <zaggy1024@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Format.h>
#include <AK/Types.h>
#include <LibVideo/DecoderError.h>

#include "Utilities.h"

#pragma once

namespace Video::VP9 {

// (8.7.1) 1D Transforms
constexpr inline i32 cos64(u8 angle)
{
    const i32 cos64_lookup[33] = { 16384, 16364, 16305, 16207, 16069, 15893, 15679, 15426, 15137, 14811, 14449, 14053, 13623, 13160, 12665, 12140, 11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005, 6270, 5520, 4756, 3981, 3196, 2404, 1606, 804, 0 };

    // 1. Set a variable angle2 equal to angle & 127.
    angle &= 127;
    // 2. If angle2 is greater than or equal to 0 and less than or equal to 32, return cos64_lookup[ angle2 ].
    if (angle <= 32)
        return cos64_lookup[angle];
    // 3. If angle2 is greater than 32 and less than or equal to 64, return cos64_lookup[ 64 - angle2 ] * -1.
    if (angle <= 64)
        return -cos64_lookup[64 - angle];
    // 4. If angle2 is greater than 64 and less than or equal to 96, return cos64_lookup[ angle2 - 64 ] * -1.
    if (angle <= 96)
        return -cos64_lookup[angle - 64];
    // 5. Otherwise (if angle2 is greater than 96 and less than 128), return cos64_lookup[ 128 - angle2 ].
    return cos64_lookup[128 - angle];
}

constexpr inline i32 sin64(u8 angle)
{
    if (angle < 32)
        angle += 128;
    return cos64(angle - 32u);
}

// (8.7.1.1) The function B( a, b, angle, 0 ) performs a butterfly rotation.
// This implementation requires that input and output do not alias.
template<typename S, typename T>
ALWAYS_INLINE void butterfly_rotation_and_rounding(T* output, size_t out_index_a, size_t out_index_b, S const* input, size_t in_index_a, size_t in_index_b, u8 angle)
{
    T cos = cos64(angle);
    T sin = sin64(angle);
    // 1. The variable x is set equal to T[ a ] * cos64( angle ) - T[ b ] * sin64( angle ).
    // 3. T[ a ] is set equal to Round2( x, 14 ).
    output[out_index_a] = rounded_right_shift(input[in_index_a] * cos - input[in_index_b] * sin, 14);
    // 2. The variable y is set equal to T[ a ] * sin64( angle ) + T[ b ] * cos64( angle ).
    // 4. T[ b ] is set equal to Round2( y, 14 ).
    output[out_index_b] = rounded_right_shift(input[in_index_a] * sin + input[in_index_b] * cos, 14);
}

// (8.7.1.1) The function B( a, b, angle, 0 ) performs a butterfly rotation.
// This implementation requires that input and output do not alias.
template<typename S, typename T>
ALWAYS_INLINE void butterfly_rotation_and_rounding(T* output, S const* input, size_t index_a, size_t index_b, u8 angle, bool flip)
{
    if (!flip)
        butterfly_rotation_and_rounding(output, index_a, index_b, input, index_a, index_b, angle);
    else
        butterfly_rotation_and_rounding(output, index_b, index_a, input, index_a, index_b, angle);
}

// (8.7.1.1) The function B( a, b, angle, 0 ) performs a butterfly rotation.
template<typename T>
ALWAYS_INLINE void butterfly_rotation_and_rounding_in_place(T* data, size_t index_a, size_t index_b, u8 angle, bool flip)
{
    Array<T, 2> temp;
    if (!flip)
        butterfly_rotation_and_rounding(temp.data(), 0, 1, data, index_a, index_b, angle);
    else
        butterfly_rotation_and_rounding(temp.data(), 1, 0, data, index_a, index_b, angle);
    data[index_a] = temp[0];
    data[index_b] = temp[1];
}

// (8.7.1.1) The function H( a, b, 0 ) performs a Hadamard rotation.
// This implementation requires that input and output do not alias.
ALWAYS_INLINE void hadamard_rotation(auto* output, size_t out_index_a, size_t out_index_b, auto const* input, size_t in_index_a, size_t in_index_b)
{
    // The function H( a, b, 0 ) performs a Hadamard rotation specified by the following ordered steps:

    // 1. The variable x is set equal to T[ a ].
    // 2. The variable y is set equal to T[ b ].
    // 3. T[ a ] is set equal to x + y.
    output[out_index_a] = input[in_index_a] + input[in_index_b];
    // 4. T[ b ] is set equal to x - y.
    output[out_index_b] = input[in_index_a] - input[in_index_b];
}

// (8.7.1.1) The function H( a, b, 0 ) performs a Hadamard rotation.
// This implementation requires that input and output do not alias.
ALWAYS_INLINE void hadamard_rotation(auto* output, auto const* input, size_t index_a, size_t index_b)
{
    hadamard_rotation(output, index_a, index_b, input, index_a, index_b);
}

// (8.7.1.1) The function B( a, b, angle, 0 ) performs a butterfly rotation.
template<typename T>
ALWAYS_INLINE void hadamard_rotation_in_place(T* data, size_t index_a, size_t index_b)
{
    Array<T, 2> temp;
    hadamard_rotation(temp.data(), 0, 1, data, index_a, index_b);
    data[index_a] = temp[0];
    data[index_b] = temp[1];
}

// 8.7.1.2 Inverse DCT array permutation process
// This process performs an in-place permutation of the array T of length 2 n for 2 ≤ n ≤ 5 which is required before
// execution of the inverse DCT process.
//
// The input to this process is a variable n that specifies the base 2 logarithm of the length of the input array.
// A temporary array named copyT is set equal to T.
//
// T[ i ] is set equal to copyT[ brev( n, i ) ] for i = 0..((1<<n) - 1).

// 8.7.1.3 Inverse DCT process
// This process performs an in-place inverse discrete cosine transform of the permuted array T which is of length
// 2 n for 2 ≤ n ≤ 5.
//
// The input to this process is a variable n that specifies the base 2 logarithm of the length of the input array.
// The variable n0 is set equal to 1<<n.
// The variable n1 is set equal to 1<<(n-1).
// The variable n2 is set equal to 1<<(n-2).
// The variable n3 is set equal to 1<<(n-3).
//
// The following ordered steps apply:
// 1. If n is equal to 2, invoke B( 0, 1, 16, 1 ), otherwise recursively invoke the inverse DCT defined in this
// section with the variable n set equal to n - 1.
// 2. Invoke B( n1+i, n0-1-i, 32-brev( 5, n1+i), 0 ) for i = 0..(n2-1).
// 3. If n is greater than or equal to 3:
//     a. Invoke H( n1+4*i+2*j, n1+1+4*i+2*j, j ) for i = 0..(n3-1), j = 0..1.
// 4. If n is equal to 5:
//     a. Invoke B( n0-n+3-n2*j-4*i, n1+n-4+n2*j+4*i, 28-16*i+56*j, 1 ) for i = 0..1, j = 0..1.
//     b. Invoke H( n1+n3*j+i, n1+n2-5+n3*j-i, j&1 ) for i = 0..1, j = 0..3.
// 5. If n is greater than or equal to 4:
//     a. Invoke B( n0-n+2-i-n2*j, n1+n-3+i+n2*j, 24+48*j, 1 ) for i = 0..(n==5), j = 0..1.
//     b. Invoke H( n1+n2*j+i, n1+n2-1+n2*j-i, j&1 ) for i = 0..(2n-7), j = 0..1.
// 6. If n is greater than or equal to 3:
//     a. Invoke B( n0-n3-1-i, n1+n3+i, 16, 1 ) for i = 0..(n3-1).
// 7. Invoke H( i, n0-1-i, 0 ) for i = 0..(n1-1).

// OPTIMIZATION: The steps for inverse DCTs have been unrolled into individual operations, flattening recursion. This allows us
// to reorder the operations according to their dependence on previous values, making use of temporary arrays to parallelize them
// better. Clang does a decent job of vectorizing these transforms when they are inlined into the loops that call them.
//
// However, it would be good to write actual SIMD vector versions of these transforms to hopefully make better use of the SIMD
// registers. Currently, Clang will not vectorize the smaller sets of butterfly rotations, which may not be good for performance.
// If we can instead create constexpr vectors of the cos/sin coefficients that are used in each set of operations, then perhaps
// it would perform better, since it will not need to broadcast or swizzle scalars in the vectors.
//
// Since the transforms are flattened instead of recursive on the decreasing block size, the transforms will have inline comments
// saying "Step x - Size y", where the step refers to one defined in 8.7.1.3 above, and the size refers to the `n0` variable's value
// at the top of the recursion stack. That variable indicates the width and height of the block in pixels.

// (8.7.1.1) The function B( a, b, angle, 0 ) performs a butterfly rotation.
// This implementation requires that input and output do not alias.
// The input indices will have a bit reversal of `transform_size` applied.
template<u8 transform_size>
ALWAYS_INLINE void inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding(auto* output, auto const* input, size_t index_a, size_t index_b, u8 angle, bool flip)
{
    auto in_index_a = brev<transform_size>(index_a);
    auto in_index_b = brev<transform_size>(index_b);
    if (!flip)
        butterfly_rotation_and_rounding(output, index_a, index_b, input, in_index_a, in_index_b, angle);
    else
        butterfly_rotation_and_rounding(output, index_b, index_a, input, in_index_a, in_index_b, angle);
}

template<typename T>
ALWAYS_INLINE void inverse_discrete_cosine_transform_4(T* data)
{
    Array<T, 4> temp_1;
    // Step 1
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<2>(temp_1.data(), data, 0, 1, 16, true);
    // Step 2
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<2>(temp_1.data(), data, 2, 3, 24, false);

    // Step 7
    hadamard_rotation(data, temp_1.data(), 0, 3);
    hadamard_rotation(data, temp_1.data(), 1, 2);
}

template<typename T>
ALWAYS_INLINE void inverse_discrete_cosine_transform_8(T* data)
{
    Array<T, 8> temp_1;
    Array<T, 8> temp_2;
    //   Step 1 - Size 4
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<3>(temp_1.data(), data, 0, 1, 16, true);
    //   Step 2 - Size 4
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<3>(temp_1.data(), data, 2, 3, 24, false);
    // Step 2 - Size 8
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<3>(temp_1.data(), data, 4, 7, 28, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<3>(temp_1.data(), data, 5, 6, 12, false);

    //   Step 7 - Size 4
    hadamard_rotation(temp_2.data(), temp_1.data(), 0, 3);
    hadamard_rotation(temp_2.data(), temp_1.data(), 1, 2);
    // Step 3 - Size 8
    hadamard_rotation(temp_2.data(), temp_1.data(), 4, 5);
    hadamard_rotation(temp_2.data(), temp_1.data(), 7, 6);

    memcpy(temp_1.data(), temp_2.data(), sizeof(temp_1));
    // Step 6 - Size 8
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 6, 5, 16, true);

    // Step 7 - Size 8
    hadamard_rotation(data, temp_1.data(), 0, 7);
    hadamard_rotation(data, temp_1.data(), 1, 6);
    hadamard_rotation(data, temp_1.data(), 2, 5);
    hadamard_rotation(data, temp_1.data(), 3, 4);
}

template<typename T>
ALWAYS_INLINE void inverse_discrete_cosine_transform_16(T* data)
{
    Array<T, 16> temp_1;
    Array<T, 16> temp_2;

    //     Step 1 - Size 4
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 0, 1, 16, true);
    //     Step 2 - Size 4
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 2, 3, 24, false);
    //   Step 2 - Size 8
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 4, 7, 28, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 5, 6, 12, false);
    // Step 2 - Size 16
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 8, 15, 30, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 9, 14, 14, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 10, 13, 22, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<4>(temp_1.data(), data, 11, 12, 6, false);

    //     Step 7 - Size 4
    hadamard_rotation(temp_2.data(), temp_1.data(), 0, 3);
    hadamard_rotation(temp_2.data(), temp_1.data(), 1, 2);
    //   Step 3 - Size 8
    hadamard_rotation(temp_2.data(), temp_1.data(), 4, 5);
    hadamard_rotation(temp_2.data(), temp_1.data(), 7, 6);
    // Step 3 - Size 16
    hadamard_rotation(temp_2.data(), temp_1.data(), 8, 9);
    hadamard_rotation(temp_2.data(), temp_1.data(), 11, 10);
    hadamard_rotation(temp_2.data(), temp_1.data(), 12, 13);
    hadamard_rotation(temp_2.data(), temp_1.data(), 15, 14);

    memcpy(temp_1.data(), temp_2.data(), sizeof(temp_1));
    //   Step 6 - Size 8
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 6, 5, 16, true);
    // Step 5a - Size 16
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 14, 9, 24, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 10, 13, 72, true);

    //   Step 7 - Size 8
    hadamard_rotation(temp_2.data(), temp_1.data(), 0, 7);
    hadamard_rotation(temp_2.data(), temp_1.data(), 1, 6);
    hadamard_rotation(temp_2.data(), temp_1.data(), 2, 5);
    hadamard_rotation(temp_2.data(), temp_1.data(), 3, 4);
    // Step 5b - Size 16
    hadamard_rotation(temp_2.data(), temp_1.data(), 8, 11);
    hadamard_rotation(temp_2.data(), temp_1.data(), 15, 12);
    hadamard_rotation(temp_2.data(), temp_1.data(), 9, 10);
    hadamard_rotation(temp_2.data(), temp_1.data(), 14, 13);

    memcpy(temp_1.data(), temp_2.data(), sizeof(temp_1));
    // Step 6 - Size 16
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 13, 10, 16, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 12, 11, 16, true);

    // Step 7 - Size 16
    hadamard_rotation(data, temp_1.data(), 0, 15);
    hadamard_rotation(data, temp_1.data(), 1, 14);
    hadamard_rotation(data, temp_1.data(), 2, 13);
    hadamard_rotation(data, temp_1.data(), 3, 12);
    hadamard_rotation(data, temp_1.data(), 4, 11);
    hadamard_rotation(data, temp_1.data(), 5, 10);
    hadamard_rotation(data, temp_1.data(), 6, 9);
    hadamard_rotation(data, temp_1.data(), 7, 8);
}

template<typename T>
ALWAYS_INLINE void inverse_discrete_cosine_transform_32(T* data)
{
    Array<T, 32> temp_1;
    Array<T, 32> temp_2;

    //       Step 1 - Size 4
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 0, 1, 16, true);
    //       Step 2 - Size 4
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 2, 3, 24, false);
    //     Step 2 - Size 8
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 4, 7, 28, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 5, 6, 12, false);
    //   Step 2 - Size 16
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 8, 15, 30, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 9, 14, 14, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 10, 13, 22, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 11, 12, 6, false);
    // Step 2 - Size 32
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 16, 31, 31, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 17, 30, 15, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 18, 29, 23, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 19, 28, 7, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 20, 27, 27, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 21, 26, 11, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 22, 25, 19, false);
    inverse_discrete_cosine_transform_input_butterfly_rotation_and_rounding<5>(temp_1.data(), data, 23, 24, 3, false);

    //       Step 7 - Size 4
    hadamard_rotation(temp_2.data(), temp_1.data(), 0, 3);
    hadamard_rotation(temp_2.data(), temp_1.data(), 1, 2);
    //     Step 3 - Size 8
    hadamard_rotation(temp_2.data(), temp_1.data(), 4, 5);
    hadamard_rotation(temp_2.data(), temp_1.data(), 7, 6);
    //   Step 3 - Size 16
    hadamard_rotation(temp_2.data(), temp_1.data(), 8, 9);
    hadamard_rotation(temp_2.data(), temp_1.data(), 11, 10);
    hadamard_rotation(temp_2.data(), temp_1.data(), 12, 13);
    hadamard_rotation(temp_2.data(), temp_1.data(), 15, 14);
    // Step 3 - Size 32
    hadamard_rotation(temp_2.data(), temp_1.data(), 16, 17);
    hadamard_rotation(temp_2.data(), temp_1.data(), 19, 18);
    hadamard_rotation(temp_2.data(), temp_1.data(), 20, 21);
    hadamard_rotation(temp_2.data(), temp_1.data(), 23, 22);
    hadamard_rotation(temp_2.data(), temp_1.data(), 24, 25);
    hadamard_rotation(temp_2.data(), temp_1.data(), 27, 26);
    hadamard_rotation(temp_2.data(), temp_1.data(), 28, 29);
    hadamard_rotation(temp_2.data(), temp_1.data(), 31, 30);

    memcpy(temp_1.data(), temp_2.data(), sizeof(temp_1));
    //     Step 6 - Size 8
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 6, 5, 16, true);
    //   Step 5a - Size 16
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 14, 9, 24, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 10, 13, 72, true);
    // Step 4a - Size 32
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 30, 17, 28, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 22, 25, 84, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 26, 21, 12, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 18, 29, 68, true);

    //     Step 7 - Size 8
    hadamard_rotation(temp_2.data(), temp_1.data(), 0, 7);
    hadamard_rotation(temp_2.data(), temp_1.data(), 1, 6);
    hadamard_rotation(temp_2.data(), temp_1.data(), 2, 5);
    hadamard_rotation(temp_2.data(), temp_1.data(), 3, 4);
    //   Step 5b - Size 16
    hadamard_rotation(temp_2.data(), temp_1.data(), 8, 11);
    hadamard_rotation(temp_2.data(), temp_1.data(), 15, 12);
    hadamard_rotation(temp_2.data(), temp_1.data(), 9, 10);
    hadamard_rotation(temp_2.data(), temp_1.data(), 14, 13);
    // Step 4b - Size 32
    hadamard_rotation(temp_2.data(), temp_1.data(), 16, 19);
    hadamard_rotation(temp_2.data(), temp_1.data(), 23, 20);
    hadamard_rotation(temp_2.data(), temp_1.data(), 24, 27);
    hadamard_rotation(temp_2.data(), temp_1.data(), 31, 28);
    hadamard_rotation(temp_2.data(), temp_1.data(), 17, 18);
    hadamard_rotation(temp_2.data(), temp_1.data(), 22, 21);
    hadamard_rotation(temp_2.data(), temp_1.data(), 25, 26);
    hadamard_rotation(temp_2.data(), temp_1.data(), 30, 29);

    memcpy(temp_1.data(), temp_2.data(), sizeof(temp_1));
    //   Step 6 - Size 16
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 13, 10, 16, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 12, 11, 16, true);
    // Step 5a - Size 32
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 29, 18, 24, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 21, 26, 72, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 28, 19, 24, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 20, 27, 72, true);

    //   Step 7 - Size 16
    hadamard_rotation(temp_2.data(), temp_1.data(), 0, 15);
    hadamard_rotation(temp_2.data(), temp_1.data(), 1, 14);
    hadamard_rotation(temp_2.data(), temp_1.data(), 2, 13);
    hadamard_rotation(temp_2.data(), temp_1.data(), 3, 12);
    hadamard_rotation(temp_2.data(), temp_1.data(), 4, 11);
    hadamard_rotation(temp_2.data(), temp_1.data(), 5, 10);
    hadamard_rotation(temp_2.data(), temp_1.data(), 6, 9);
    hadamard_rotation(temp_2.data(), temp_1.data(), 7, 8);
    // Step 5b - Size 32
    hadamard_rotation(temp_2.data(), temp_1.data(), 16, 23);
    hadamard_rotation(temp_2.data(), temp_1.data(), 31, 24);
    hadamard_rotation(temp_2.data(), temp_1.data(), 17, 22);
    hadamard_rotation(temp_2.data(), temp_1.data(), 30, 25);
    hadamard_rotation(temp_2.data(), temp_1.data(), 18, 21);
    hadamard_rotation(temp_2.data(), temp_1.data(), 29, 26);
    hadamard_rotation(temp_2.data(), temp_1.data(), 19, 20);
    hadamard_rotation(temp_2.data(), temp_1.data(), 28, 27);

    memcpy(temp_1.data(), temp_2.data(), sizeof(temp_1));
    // Step 6 - Size 32
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 27, 20, 16, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 26, 21, 16, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 25, 22, 16, true);
    butterfly_rotation_and_rounding(temp_1.data(), temp_2.data(), 24, 23, 16, true);

    // Step 7 - Size 32
    hadamard_rotation(data, temp_1.data(), 0, 31);
    hadamard_rotation(data, temp_1.data(), 1, 30);
    hadamard_rotation(data, temp_1.data(), 2, 29);
    hadamard_rotation(data, temp_1.data(), 3, 28);
    hadamard_rotation(data, temp_1.data(), 4, 27);
    hadamard_rotation(data, temp_1.data(), 5, 26);
    hadamard_rotation(data, temp_1.data(), 6, 25);
    hadamard_rotation(data, temp_1.data(), 7, 24);
    hadamard_rotation(data, temp_1.data(), 8, 23);
    hadamard_rotation(data, temp_1.data(), 9, 22);
    hadamard_rotation(data, temp_1.data(), 10, 21);
    hadamard_rotation(data, temp_1.data(), 11, 20);
    hadamard_rotation(data, temp_1.data(), 12, 19);
    hadamard_rotation(data, temp_1.data(), 13, 18);
    hadamard_rotation(data, temp_1.data(), 14, 17);
    hadamard_rotation(data, temp_1.data(), 15, 16);
}

// This is the simplification of the above inverse DCTs, but when `data`
// only contains one non-zero coefficient at index 0.
template<u8 log2_of_block_size>
static ALWAYS_INLINE void inverse_discrete_cosine_transform_1_coef(auto* data)
{
    // Note that this transform is flipped, so we put sin first.
    constexpr auto block_size = 1 << log2_of_block_size;
    auto dc_value = rounded_right_shift(data[0] * sin64(16), 14);

    for (u8 i = 0; i < block_size; i++)
        data[i] = dc_value;
}

// (8.7.1.3) Inverse DCT process
template<u8 log2_of_block_size, typename T>
ALWAYS_INLINE void inverse_discrete_cosine_transform(T* data)
{
    static_assert(log2_of_block_size >= 2 && log2_of_block_size <= 5, "Block size out of range.");

    if constexpr (log2_of_block_size == 2) {
        inverse_discrete_cosine_transform_4(data);
    } else if constexpr (log2_of_block_size == 3) {
        inverse_discrete_cosine_transform_8(data);
    } else if constexpr (log2_of_block_size == 4) {
        inverse_discrete_cosine_transform_16(data);
    } else if constexpr (log2_of_block_size == 5) {
        inverse_discrete_cosine_transform_32(data);
    } else {
        static_assert("IDCT transform size is not allowed.");
    }
}

template<u8 log2_of_block_size, typename T>
inline void inverse_asymmetric_discrete_sine_transform_input_array_permutation(T* data)
{
    // The variable n0 is set equal to 1<<n.
    constexpr auto block_size = 1u << log2_of_block_size;
    // The variable n1 is set equal to 1<<(n-1).
    // We can iterate by 2 at a time instead of taking half block size.

    // A temporary array named copyT is set equal to T.
    Array<T, block_size> data_copy;
    AK::TypedTransfer<T>::copy(data_copy.data(), data, block_size);

    // The values at even locations T[ 2 * i ] are set equal to copyT[ n0 - 1 - 2 * i ] for i = 0..(n1-1).
    // The values at odd locations T[ 2 * i + 1 ] are set equal to copyT[ 2 * i ] for i = 0..(n1-1).
    for (auto i = 0u; i < block_size; i += 2) {
        data[i] = data_copy[block_size - 1 - i];
        data[i + 1] = data_copy[i];
    }
}

// (8.7.1.6) This process does an in-place transform of the array T to perform an inverse ADST.
template<typename T>
inline void inverse_asymmetric_discrete_sine_transform_4(T* data)
{
    const i64 sinpi_1_9 = 5283;
    const i64 sinpi_2_9 = 9929;
    const i64 sinpi_3_9 = 13377;
    const i64 sinpi_4_9 = 15212;

    // Steps are derived from pseudocode in (8.7.1.6):
    // s0 = SINPI_1_9 * T[ 0 ]
    i64 s0 = sinpi_1_9 * data[0];
    // s1 = SINPI_2_9 * T[ 0 ]
    i64 s1 = sinpi_2_9 * data[0];
    // s2 = SINPI_3_9 * T[ 1 ]
    i64 s2 = sinpi_3_9 * data[1];
    // s3 = SINPI_4_9 * T[ 2 ]
    i64 s3 = sinpi_4_9 * data[2];
    // s4 = SINPI_1_9 * T[ 2 ]
    i64 s4 = sinpi_1_9 * data[2];
    // s5 = SINPI_2_9 * T[ 3 ]
    i64 s5 = sinpi_2_9 * data[3];
    // s6 = SINPI_4_9 * T[ 3 ]
    i64 s6 = sinpi_4_9 * data[3];
    // v = T[ 0 ] - T[ 2 ] + T[ 3 ]
    // s7 = SINPI_3_9 * v
    i64 s7 = sinpi_3_9 * (data[0] - data[2] + data[3]);

    // x0 = s0 + s3 + s5
    auto x0 = s0 + s3 + s5;
    // x1 = s1 - s4 - s6
    auto x1 = s1 - s4 - s6;
    // x2 = s7
    auto x2 = s7;
    // x3 = s2
    auto x3 = s2;

    // s0 = x0 + x3
    s0 = x0 + x3;
    // s1 = x1 + x3
    s1 = x1 + x3;
    // s2 = x2
    s2 = x2;
    // s3 = x0 + x1 - x3
    s3 = x0 + x1 - x3;

    // T[ 0 ] = Round2( s0, 14 )
    data[0] = rounded_right_shift(s0, 14);
    // T[ 1 ] = Round2( s1, 14 )
    data[1] = rounded_right_shift(s1, 14);
    // T[ 2 ] = Round2( s2, 14 )
    data[2] = rounded_right_shift(s2, 14);
    // T[ 3 ] = Round2( s3, 14 )
    data[3] = rounded_right_shift(s3, 14);

    // (8.7.1.1) The inverse asymmetric discrete sine transforms also make use of an T array named S.
    // The values in this array require higher precision to avoid overflow. Using signed integers with 24 +
    // BitDepth bits of precision is enough to avoid overflow.
    // Note: Since bounds checks just ensure that we will not have resulting values that will overflow, it's non-fatal
    // to allow these bounds to be violated. Therefore, we can avoid the performance cost here.
}

// The function SB( a, b, angle, 0 ) performs a butterfly rotation.
// Spec defines the source as array T, and the destination array as S.
template<typename S, typename T>
ALWAYS_INLINE void butterfly_rotation(T* destination, u8 out_index_a, u8 out_index_b, S* source, u8 in_index_a, u8 in_index_b, u8 angle)
{
    // The function SB( a, b, angle, 0 ) performs a butterfly rotation according to the following ordered steps:
    T cos = cos64(angle);
    T sin = sin64(angle);
    // 1. S[ a ] is set equal to T[ a ] * cos64( angle ) - T[ b ] * sin64( angle ).
    destination[out_index_a] = source[in_index_a] * cos - source[in_index_b] * sin;
    // 2. S[ b ] is set equal to T[ a ] * sin64( angle ) + T[ b ] * cos64( angle ).
    destination[out_index_b] = source[in_index_a] * sin + source[in_index_b] * cos;
}

// The function SB( a, b, angle, 0 ) performs a butterfly rotation.
// Spec defines the source as array T, and the destination array as S.
template<typename S, typename T>
ALWAYS_INLINE void butterfly_rotation(T* destination, S* source, u8 index_a, u8 index_b, u8 angle, bool flip)
{
    if (!flip)
        butterfly_rotation(destination, index_a, index_b, source, index_a, index_b, angle);
    else
        butterfly_rotation(destination, index_b, index_a, source, index_a, index_b, angle);
}

// The function SH( a, b ) performs a Hadamard rotation and rounding.
// Spec defines the source array as S, and the destination array as T.
template<typename S, typename T>
ALWAYS_INLINE void hadamard_rotation_and_rounding(S* source, T* destination, size_t index_a, size_t index_b)
{
    // 1. T[ a ] is set equal to Round2( S[ a ] + S[ b ], 14 ).
    destination[index_a] = rounded_right_shift(source[index_a] + source[index_b], 14);
    // 2. T[ b ] is set equal to Round2( S[ a ] - S[ b ], 14 ).
    destination[index_b] = rounded_right_shift(source[index_a] - source[index_b], 14);
}

// Applies inverse ADST output permutations and negations without branching.
template<u8 log2_of_block_size, typename T>
ALWAYS_INLINE void inverse_asymmetric_discrete_sine_transform_output(T* destination, T* source)
{
    constexpr auto block_size = 1 << log2_of_block_size;
    for (size_t to_index = 0; to_index < block_size; to_index++) {
        // − If n is equal to 4,
        // T[ 8*a + 4*b + 2*c + d ] is set equal to copyT[ 8*(d^c) + 4*(c^b) + 2*(b^a) + a ] for a = 0..1
        // and b = 0..1 and c = 0..1 and d = 0..1.
        // − Otherwise (n is equal to 3),
        // T[ 4*a + 2*b + c ] is set equal to copyT[ 4*(c^b) + 2*(b^a) + a ] for a = 0..1 and
        // b = 0..1 and c = 0..1.

        // The loops above boil down to (i = brev(i); i ^= (i << 1) & ).
        auto from_index = brev<log2_of_block_size>(to_index);
        constexpr auto max_index = (1 << log2_of_block_size) - 1;
        from_index ^= (from_index << 1) & max_index;

        // - If n is equal to 4,
        // Set T[ 1+12*j+2*i ] equal to -T[ 1+12*j+2*i ] for i = 0..1, for j = 0..1.
        // - Otherwise,
        // Set T[ 1+2*i ] equal to -T[ 1+2*i ] for i = 0..3.

        // These two loops have in common that they are negating the two first and last
        // uneven indices.
        auto negate = to_index == 1 || to_index == 3 || to_index == max_index || to_index == max_index - 2;
        destination[to_index] = (source[from_index] ^ -negate) + negate;
    }
}

// (8.7.1.7) This process does an in-place transform of the array T using a higher precision array S for intermediate
// results.
template<typename T>
ALWAYS_INLINE void inverse_asymmetric_discrete_sine_transform_8(T* data)
{
    // This process does an in-place transform of the array T using:

    // A higher precision array S for T results.
    // (8.7.1.1) NOTE - The values in array S require higher precision to avoid overflow. Using signed integers with
    // 24 + BitDepth bits of precision is enough to avoid overflow.
    Array<i64, 8> high_precision_temp;

    // The following ordered steps apply:

    // 1. Invoke the ADST input array permutation process specified in section 8.7.1.4 with the input variable n set
    //    equal to 3.
    inverse_asymmetric_discrete_sine_transform_input_array_permutation<3>(data);

    // 2. Invoke SB( 2*i, 1+2*i, 30-8*i, 1 ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        butterfly_rotation(high_precision_temp.data(), data, 2 * i, 1 + (2 * i), 30 - (8 * i), true);

    // 3. Invoke SH( i, 4+i ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        hadamard_rotation_and_rounding(high_precision_temp.data(), data, i, 4 + i);

    // 4. Invoke SB( 4+3*i, 5+i, 24-16*i, 1 ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        butterfly_rotation(high_precision_temp.data(), data, 4 + (3 * i), 5 + i, 24 - (16 * i), true);
    // 5. Invoke SH( 4+i, 6+i ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        hadamard_rotation_and_rounding(high_precision_temp.data(), data, 4 + i, 6 + i);

    // 6. Invoke H( i, 2+i, 0 ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        hadamard_rotation_in_place(data, i, 2 + i);

    // 7. Invoke B( 2+4*i, 3+4*i, 16, 1 ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        butterfly_rotation_and_rounding_in_place(data, 2 + (4 * i), 3 + (4 * i), 16, true);

    // 8. Invoke the ADST output array permutation process specified in section 8.7.1.5 with the input variable n
    //    set equal to 3.
    // 9. Set T[ 1+2*i ] equal to -T[ 1+2*i ] for i = 0..3.
    Array<T, 8> data_copy;
    memcpy(data_copy.data(), data, sizeof(data_copy));
    inverse_asymmetric_discrete_sine_transform_output<3>(data, data_copy.data());
}

// (8.7.1.8) This process does an in-place transform of the array T using a higher precision array S for intermediate
// results.
template<typename T>
ALWAYS_INLINE void inverse_asymmetric_discrete_sine_transform_16(T* data)
{
    // This process does an in-place transform of the array T using:

    // A higher precision array S for T results.
    // (8.7.1.1) The inverse asymmetric discrete sine transforms also make use of an T array named S.
    // The values in this array require higher precision to avoid overflow. Using signed integers with 24 +
    // BitDepth bits of precision is enough to avoid overflow.
    Array<i64, 16> high_precision_temp;

    // The following ordered steps apply:

    // 1. Invoke the ADST input array permutation process specified in section 8.7.1.4 with the input variable n set
    // equal to 4.
    inverse_asymmetric_discrete_sine_transform_input_array_permutation<4>(data);

    // 2. Invoke SB( 2*i, 1+2*i, 31-4*i, 1 ) for i = 0..7.
    for (auto i = 0u; i < 8; i++)
        butterfly_rotation(high_precision_temp.data(), data, 2 * i, 1 + (2 * i), 31 - (4 * i), true);
    // 3. Invoke SH( i, 8+i ) for i = 0..7.
    for (auto i = 0u; i < 8; i++)
        hadamard_rotation_and_rounding(high_precision_temp.data(), data, i, 8 + i);

    // 4. Invoke SB( 8+2*i, 9+2*i, 28-16*i, 1 ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        butterfly_rotation(high_precision_temp.data(), data, 8 + (2 * i), 9 + (2 * i), 128 + 28 - (16 * i), true);
    // 5. Invoke SH( 8+i, 12+i ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        hadamard_rotation_and_rounding(high_precision_temp.data(), data, 8 + i, 12 + i);

    // 6. Invoke H( i, 4+i, 0 ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        hadamard_rotation_in_place(data, i, 4 + i);

    // 7. Invoke SB( 4+8*i+3*j, 5+8*i+j, 24-16*j, 1 ) for i = 0..1, for j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            butterfly_rotation(high_precision_temp.data(), data, 4 + (8 * i) + (3 * j), 5 + (8 * i) + j, 24 - (16 * j), true);
    // 8. Invoke SH( 4+8*j+i, 6+8*j+i ) for i = 0..1, j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            hadamard_rotation_and_rounding(high_precision_temp.data(), data, 4 + (8 * j) + i, 6 + (8 * j) + i);

    // 9. Invoke H( 8*j+i, 2+8*j+i, 0 ) for i = 0..1, for j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            hadamard_rotation_in_place(data, (8 * j) + i, 2 + (8 * j) + i);
    // 10. Invoke B( 2+4*j+8*i, 3+4*j+8*i, 48+64*(i^j), 0 ) for i = 0..1, for j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            butterfly_rotation_and_rounding_in_place(data, 2 + (4 * j) + (8 * i), 3 + (4 * j) + (8 * i), 48 + (64 * (i ^ j)), false);

    // 11. Invoke the ADST output array permutation process specified in section 8.7.1.5 with the input variable n
    // set equal to 4.
    // 12. Set T[ 1+12*j+2*i ] equal to -T[ 1+12*j+2*i ] for i = 0..1, for j = 0..1.
    Array<T, 16> data_copy;
    memcpy(data_copy.data(), data, sizeof(data_copy));
    inverse_asymmetric_discrete_sine_transform_output<4>(data, data_copy.data());
}

template<u8 log2_of_block_size, typename T>
ALWAYS_INLINE void inverse_asymmetric_discrete_sine_transform(T* data)
{
    // 8.7.1.9 Inverse ADST Process
    // This process performs an in-place inverse ADST process on the array T of size 2^n for 2 ≤ n ≤ 4.

    // The process to invoke depends on n as follows:
    if constexpr (log2_of_block_size == 2) {
        // − If n is equal to 2, invoke the Inverse ADST4 process specified in section 8.7.1.6.
        inverse_asymmetric_discrete_sine_transform_4(data);
    } else if constexpr (log2_of_block_size == 3) {
        // − Otherwise if n is equal to 3, invoke the Inverse ADST8 process specified in section 8.7.1.7.
        inverse_asymmetric_discrete_sine_transform_8(data);
        return;
    } else if constexpr (log2_of_block_size == 4) {
        // − Otherwise (n is equal to 4), invoke the Inverse ADST16 process specified in section 8.7.1.8.
        inverse_asymmetric_discrete_sine_transform_16(data);
    } else {
        VERIFY_NOT_REACHED();
    }
}

// This processes an 4-wide inverse ADST for a data set that has only one non-zero coefficient on the left.
template<typename T>
inline void inverse_asymmetric_discrete_sine_transform_1_coef_4(T* data)
{
    const i64 sinpi_1_9 = 5283;
    const i64 sinpi_2_9 = 9929;
    const i64 sinpi_3_9 = 13377;

    i64 s0 = sinpi_1_9 * data[0];
    i64 s1 = sinpi_2_9 * data[0];
    i64 s2 = sinpi_3_9 * data[0];
    auto s3 = s0 + s1;

    // T[ 0 ] = Round2( s0, 14 )
    data[0] = rounded_right_shift(s0, 14);
    // T[ 1 ] = Round2( s1, 14 )
    data[1] = rounded_right_shift(s1, 14);
    // T[ 2 ] = Round2( s2, 14 )
    data[2] = rounded_right_shift(s2, 14);
    // T[ 3 ] = Round2( s3, 14 )
    data[3] = rounded_right_shift(s3, 14);
}

// This processes an 8-wide inverse ADST for a data set that has only one non-zero coefficient on the left.
template<typename T>
ALWAYS_INLINE void inverse_asymmetric_discrete_sine_transform_1_coef_8(T* data)
{
    using HP = i64;

    Array<T, 8> intermediate;

    // Use high precision for cosine and sine so that when we multiply, it increases the precision
    // of the product. Otherwise, we will experience overflows/truncation.
    HP cos = cos64(30);
    HP sin = sin64(30);
    intermediate[0] = rounded_right_shift(data[0] * cos, 14);
    intermediate[1] = rounded_right_shift(-data[0] * sin, 14);

    data[0] = intermediate[0];
    data[7] = intermediate[1];
    cos = cos64(16);
    sin = sin64(16);
    intermediate[2] = rounded_right_shift(intermediate[0] * sin + intermediate[1] * cos, 14);
    intermediate[3] = rounded_right_shift(intermediate[0] * cos - intermediate[1] * sin, 14);
    cos = cos64(24);
    sin = sin64(24);
    intermediate[4] = rounded_right_shift(intermediate[0] * sin + intermediate[1] * cos, 14);
    intermediate[5] = rounded_right_shift(intermediate[0] * cos - intermediate[1] * sin, 14);
    cos = cos64(16);
    sin = sin64(16);
    intermediate[6] = rounded_right_shift(intermediate[4] * sin + intermediate[5] * cos, 14);
    intermediate[7] = rounded_right_shift(intermediate[4] * cos - intermediate[5] * sin, 14);

    inverse_asymmetric_discrete_sine_transform_output<3>(data, intermediate.data());
}

template<typename T>
ALWAYS_INLINE void inverse_asymmetric_discrete_sine_transform_1_coef_16(T* data)
{
    using HP = i64;

    Array<T, 16> intermediate;

    HP cos = cos64(31);
    HP sin = sin64(31);
    T s0 = rounded_right_shift(data[0] * cos, 14);
    T s1 = rounded_right_shift(-data[0] * sin, 14);
    intermediate[0] = s0;
    intermediate[1] = s1;

    cos = cos64(48);
    sin = sin64(48);
    T a = rounded_right_shift(s0 * cos - s1 * sin, 14);
    T b = rounded_right_shift(s0 * sin + s1 * cos, 14);
    intermediate[2] = a;
    intermediate[3] = b;

    cos = cos64(24);
    sin = sin64(24);
    T s5 = rounded_right_shift(s0 * cos - s1 * sin, 14);
    T s4 = rounded_right_shift(s0 * sin + s1 * cos, 14);
    intermediate[4] = s4;
    intermediate[5] = s5;

    cos = cos64(112);
    sin = sin64(112);
    a = rounded_right_shift(s4 * cos - s5 * sin, 14);
    b = rounded_right_shift(s4 * sin + s5 * cos, 14);
    intermediate[6] = a;
    intermediate[7] = b;

    cos = cos64(28);
    sin = sin64(28);
    T s9 = rounded_right_shift(s0 * cos - s1 * sin, 14);
    T s8 = rounded_right_shift(s0 * sin + s1 * cos, 14);
    intermediate[8] = s0;
    intermediate[9] = s9;

    cos = cos64(112);
    sin = sin64(112);
    a = rounded_right_shift(s0 * cos - s9 * sin, 14);
    b = rounded_right_shift(s0 * sin + s9 * cos, 14);
    intermediate[10] = a;
    intermediate[11] = b;

    cos = cos64(24);
    sin = sin64(24);
    T s13 = rounded_right_shift(s8 * cos - s9 * sin, 14);
    T s12 = rounded_right_shift(s8 * sin + s9 * cos, 14);
    intermediate[12] = s12;
    intermediate[13] = s13;

    cos = cos64(48);
    sin = sin64(48);
    a = rounded_right_shift(s12 * cos - s13 * sin, 14);
    b = rounded_right_shift(s12 * sin + s13 * cos, 14);
    intermediate[14] = a;
    intermediate[15] = b;

    inverse_asymmetric_discrete_sine_transform_output<4>(data, intermediate.data());
}

// This processes an inverse ADST for a data set that has only one non-zero coefficient on the left.
template<u8 log2_of_block_size, typename T>
ALWAYS_INLINE void inverse_asymmetric_discrete_sine_transform_1_coef(T* data)
{
    // 8.7.1.9 Inverse ADST Process
    // This process performs an in-place inverse ADST process on the array T of size 2^n for 2 ≤ n ≤ 4.

    // The process to invoke depends on n as follows:
    if constexpr (log2_of_block_size == 2) {
        // − If n is equal to 2, invoke the Inverse ADST4 process specified in section 8.7.1.6.
        inverse_asymmetric_discrete_sine_transform_1_coef_4(data);
    } else if constexpr (log2_of_block_size == 3) {
        // − Otherwise if n is equal to 3, invoke the Inverse ADST8 process specified in section 8.7.1.7.
        inverse_asymmetric_discrete_sine_transform_1_coef_8(data);
        return;
    } else if constexpr (log2_of_block_size == 4) {
        // − Otherwise (n is equal to 4), invoke the Inverse ADST16 process specified in section 8.7.1.8.
        inverse_asymmetric_discrete_sine_transform_1_coef_16(data);
    } else {
        VERIFY_NOT_REACHED();
    }
}

}
