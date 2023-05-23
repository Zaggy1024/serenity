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

inline i32 cos64(u8 angle)
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

inline i32 sin64(u8 angle)
{
    if (angle < 32)
        angle += 128;
    return cos64(angle - 32u);
}

// (8.7.1.1) The function B( a, b, angle, 0 ) performs a butterfly rotation.
template<typename T>
inline void butterfly_rotation_in_place(T* data, size_t index_a, size_t index_b, u8 angle, bool flip)
{
    auto cos = cos64(angle);
    auto sin = sin64(angle);
    // 1. The variable x is set equal to T[ a ] * cos64( angle ) - T[ b ] * sin64( angle ).
    i64 rotated_a = data[index_a] * cos - data[index_b] * sin;
    // 2. The variable y is set equal to T[ a ] * sin64( angle ) + T[ b ] * cos64( angle ).
    i64 rotated_b = data[index_a] * sin + data[index_b] * cos;
    // 3. T[ a ] is set equal to Round2( x, 14 ).
    data[index_a] = rounded_right_shift(rotated_a, 14);
    // 4. T[ b ] is set equal to Round2( y, 14 ).
    data[index_b] = rounded_right_shift(rotated_b, 14);

    // The function B( a ,b, angle, 1 ) performs a butterfly rotation and flip specified by the following ordered steps:
    // 1. The function B( a, b, angle, 0 ) is invoked.
    // 2. The contents of T[ a ] and T[ b ] are exchanged.
    if (flip)
        swap(data[index_a], data[index_b]);

    // It is a requirement of bitstream conformance that the values saved into the array T by this function are
    // representable by a signed integer using 8 + BitDepth bits of precision.
    // Note: Since bounds checks just ensure that we will not have resulting values that will overflow, it's non-fatal
    // to allow these bounds to be violated. Therefore, we can avoid the performance cost here.
}

// (8.7.1.1) The function H( a, b, 0 ) performs a Hadamard rotation.
template<typename T>
inline void hadamard_rotation_in_place(T* data, size_t index_a, size_t index_b, bool flip)
{
    // The function H( a, b, 1 ) performs a Hadamard rotation with flipped indices and is specified as follows:
    // 1. The function H( b, a, 0 ) is invoked.
    if (flip)
        swap(index_a, index_b);

    // The function H( a, b, 0 ) performs a Hadamard rotation specified by the following ordered steps:

    // 1. The variable x is set equal to T[ a ].
    auto a_value = data[index_a];
    // 2. The variable y is set equal to T[ b ].
    auto b_value = data[index_b];
    // 3. T[ a ] is set equal to x + y.
    data[index_a] = a_value + b_value;
    // 4. T[ b ] is set equal to x - y.
    data[index_b] = a_value - b_value;

    // It is a requirement of bitstream conformance that the values saved into the array T by this function are
    // representable by a signed integer using 8 + BitDepth bits of precision.
    // Note: Since bounds checks just ensure that we will not have resulting values that will overflow, it's non-fatal
    // to allow these bounds to be violated. Therefore, we can avoid the performance cost here.
}

template<u8 log2_of_block_size, typename T>
inline DecoderErrorOr<void> inverse_discrete_cosine_transform_array_permutation(T* data)
{
    static_assert(log2_of_block_size >= 2 && log2_of_block_size <= 5, "Block size out of range.");

    constexpr u8 block_size = 1 << log2_of_block_size;

    // This process performs an in-place permutation of the array T of length 2^n for 2 ≤ n ≤ 5 which is required before
    // execution of the inverse DCT process.
    if (log2_of_block_size < 2 || log2_of_block_size > 5)
        return DecoderError::corrupted("Block size was out of range"sv);

    // 1.1. A temporary array named copyT is set equal to T.
    Array<T, block_size> data_copy;
    AK::TypedTransfer<T>::copy(data_copy.data(), data, block_size);

    // 1.2. T[ i ] is set equal to copyT[ brev( n, i ) ] for i = 0..((1<<n) - 1).
    for (auto i = 0u; i < block_size; i++)
        data[i] = data_copy[brev<log2_of_block_size>(i)];

    return {};
}

template<u8 log2_of_block_size, typename T>
ALWAYS_INLINE DecoderErrorOr<void> inverse_discrete_cosine_transform(T* data)
{
    static_assert(log2_of_block_size >= 2 && log2_of_block_size <= 5, "Block size out of range.");

    // 2.1. The variable n0 is set equal to 1<<n.
    constexpr u8 block_size = 1 << log2_of_block_size;

    // 8.7.1.3 Inverse DCT process

    // 2.2. The variable n1 is set equal to 1<<(n-1).
    constexpr u8 half_block_size = block_size >> 1;
    // 2.3 The variable n2 is set equal to 1<<(n-2).
    constexpr u8 quarter_block_size = half_block_size >> 1;
    // 2.4 The variable n3 is set equal to 1<<(n-3).
    constexpr u8 eighth_block_size = quarter_block_size >> 1;

    // 2.5 If n is equal to 2, invoke B( 0, 1, 16, 1 ), otherwise recursively invoke the inverse DCT defined in this
    // section with the variable n set equal to n - 1.
    if constexpr (log2_of_block_size == 2)
        butterfly_rotation_in_place(data, 0, 1, 16, true);
    else
        TRY(inverse_discrete_cosine_transform<log2_of_block_size - 1>(data));

    // 2.6 Invoke B( n1+i, n0-1-i, 32-brev( 5, n1+i), 0 ) for i = 0..(n2-1).
    for (auto i = 0u; i < quarter_block_size; i++) {
        auto index = half_block_size + i;
        butterfly_rotation_in_place(data, index, block_size - 1 - i, 32 - brev<5>(index), false);
    }

    // 2.7 If n is greater than or equal to 3:
    if constexpr (log2_of_block_size >= 3) {
        // a. Invoke H( n1+4*i+2*j, n1+1+4*i+2*j, j ) for i = 0..(n3-1), j = 0..1.
        for (auto i = 0u; i < eighth_block_size; i++) {
            for (auto j = 0u; j < 2; j++) {
                auto index = half_block_size + (4 * i) + (2 * j);
                hadamard_rotation_in_place(data, index, index + 1, j);
            }
        }
    }

    // 4. If n is equal to 5:
    if constexpr (log2_of_block_size == 5) {
        // a. Invoke B( n0-n+3-n2*j-4*i, n1+n-4+n2*j+4*i, 28-16*i+56*j, 1 ) for i = 0..1, j = 0..1.
        for (auto i = 0u; i < 2; i++) {
            for (auto j = 0u; j < 2; j++) {
                auto index_a = block_size - log2_of_block_size + 3 - (quarter_block_size * j) - (4 * i);
                auto index_b = half_block_size + log2_of_block_size - 4 + (quarter_block_size * j) + (4 * i);
                auto angle = 28 - (16 * i) + (56 * j);
                butterfly_rotation_in_place(data, index_a, index_b, angle, true);
            }
        }

        // b. Invoke H( n1+n3*j+i, n1+n2-5+n3*j-i, j&1 ) for i = 0..1, j = 0..3.
        for (auto i = 0u; i < 2; i++) {
            for (auto j = 0u; j < 4; j++) {
                auto index_a = half_block_size + (eighth_block_size * j) + i;
                auto index_b = half_block_size + quarter_block_size - 5 + (eighth_block_size * j) - i;
                hadamard_rotation_in_place(data, index_a, index_b, (j & 1) != 0);
            }
        }
    }

    // 5. If n is greater than or equal to 4:
    if constexpr (log2_of_block_size >= 4) {
        // a. Invoke B( n0-n+2-i-n2*j, n1+n-3+i+n2*j, 24+48*j, 1 ) for i = 0..(n==5), j = 0..1.
        for (auto i = 0u; i <= (log2_of_block_size == 5); i++) {
            for (auto j = 0u; j < 2; j++) {
                auto index_a = block_size - log2_of_block_size + 2 - i - (quarter_block_size * j);
                auto index_b = half_block_size + log2_of_block_size - 3 + i + (quarter_block_size * j);
                butterfly_rotation_in_place(data, index_a, index_b, 24 + (48 * j), true);
            }
        }

        // b. Invoke H( n1+n2*j+i, n1+n2-1+n2*j-i, j&1 ) for i = 0..(2n-7), j = 0..1.
        for (auto i = 0u; i < (2 * log2_of_block_size) - 6u; i++) {
            for (auto j = 0u; j < 2; j++) {
                auto index_a = half_block_size + (quarter_block_size * j) + i;
                auto index_b = half_block_size + quarter_block_size - 1 + (quarter_block_size * j) - i;
                hadamard_rotation_in_place(data, index_a, index_b, (j & 1) != 0);
            }
        }
    }

    // 6. If n is greater than or equal to 3:
    if constexpr (log2_of_block_size >= 3) {
        // a. Invoke B( n0-n3-1-i, n1+n3+i, 16, 1 ) for i = 0..(n3-1).
        for (auto i = 0u; i < eighth_block_size; i++) {
            auto index_a = block_size - eighth_block_size - 1 - i;
            auto index_b = half_block_size + eighth_block_size + i;
            butterfly_rotation_in_place(data, index_a, index_b, 16, true);
        }
    }

    // 7. Invoke H( i, n0-1-i, 0 ) for i = 0..(n1-1).
    for (auto i = 0u; i < half_block_size; i++)
        hadamard_rotation_in_place(data, i, block_size - 1 - i, false);

    return {};
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

template<u8 log2_of_block_size, typename T>
inline void inverse_asymmetric_discrete_sine_transform_output_array_permutation(T* data)
{
    constexpr auto block_size = 1u << log2_of_block_size;

    // A temporary array named copyT is set equal to T.
    Array<T, block_size> data_copy;
    AK::TypedTransfer<T>::copy(data_copy.data(), data, block_size);

    // The permutation depends on n as follows:
    if (log2_of_block_size == 4) {
        // − If n is equal to 4,
        // T[ 8*a + 4*b + 2*c + d ] is set equal to copyT[ 8*(d^c) + 4*(c^b) + 2*(b^a) + a ] for a = 0..1
        // and b = 0..1 and c = 0..1 and d = 0..1.
        for (auto a = 0u; a < 2; a++)
            for (auto b = 0u; b < 2; b++)
                for (auto c = 0u; c < 2; c++)
                    for (auto d = 0u; d < 2; d++)
                        data[(8 * a) + (4 * b) + (2 * c) + d] = data_copy[8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a];
    } else {
        VERIFY(log2_of_block_size == 3);
        // − Otherwise (n is equal to 3),
        // T[ 4*a + 2*b + c ] is set equal to copyT[ 4*(c^b) + 2*(b^a) + a ] for a = 0..1 and
        // b = 0..1 and c = 0..1.
        for (auto a = 0u; a < 2; a++)
            for (auto b = 0u; b < 2; b++)
                for (auto c = 0u; c < 2; c++)
                    data[4 * a + 2 * b + c] = data_copy[4 * (c ^ b) + 2 * (b ^ a) + a];
    }
}

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
template<typename S, typename D>
inline void butterfly_rotation(S* source, D* destination, size_t index_a, size_t index_b, u8 angle, bool flip)
{
    // The function SB( a, b, angle, 0 ) performs a butterfly rotation according to the following ordered steps:
    auto cos = cos64(angle);
    auto sin = sin64(angle);
    // Expand to the destination buffer's precision.
    D a = source[index_a];
    D b = source[index_b];
    // 1. S[ a ] is set equal to T[ a ] * cos64( angle ) - T[ b ] * sin64( angle ).
    destination[index_a] = a * cos - b * sin;
    // 2. S[ b ] is set equal to T[ a ] * sin64( angle ) + T[ b ] * cos64( angle ).
    destination[index_b] = a * sin + b * cos;

    // The function SB( a, b, angle, 1 ) performs a butterfly rotation and flip according to the following ordered steps:
    // 1. The function SB( a, b, angle, 0 ) is invoked.
    // 2. The contents of S[ a ] and S[ b ] are exchanged.
    if (flip)
        swap(destination[index_a], destination[index_b]);
}

// The function SH( a, b ) performs a Hadamard rotation and rounding.
// Spec defines the source array as S, and the destination array as T.
template<typename S, typename D>
inline void hadamard_rotation(S* source, D* destination, size_t index_a, size_t index_b)
{
    // Keep the source buffer's precision until rounding.
    S a = source[index_a];
    S b = source[index_b];
    // 1. T[ a ] is set equal to Round2( S[ a ] + S[ b ], 14 ).
    destination[index_a] = rounded_right_shift(a + b, 14);
    // 2. T[ b ] is set equal to Round2( S[ a ] - S[ b ], 14 ).
    destination[index_b] = rounded_right_shift(a - b, 14);
}

template<typename T>
inline DecoderErrorOr<void> inverse_asymmetric_discrete_sine_transform_8(T* data)
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
        butterfly_rotation(data, high_precision_temp.data(), 2 * i, 1 + (2 * i), 30 - (8 * i), true);

    // 3. Invoke SH( i, 4+i ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        hadamard_rotation(high_precision_temp.data(), data, i, 4 + i);

    // 4. Invoke SB( 4+3*i, 5+i, 24-16*i, 1 ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        butterfly_rotation(data, high_precision_temp.data(), 4 + (3 * i), 5 + i, 24 - (16 * i), true);
    // 5. Invoke SH( 4+i, 6+i ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        hadamard_rotation(high_precision_temp.data(), data, 4 + i, 6 + i);

    // 6. Invoke H( i, 2+i, 0 ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        hadamard_rotation_in_place(data, i, 2 + i, false);

    // 7. Invoke B( 2+4*i, 3+4*i, 16, 1 ) for i = 0..1.
    for (auto i = 0u; i < 2; i++)
        butterfly_rotation_in_place(data, 2 + (4 * i), 3 + (4 * i), 16, true);

    // 8. Invoke the ADST output array permutation process specified in section 8.7.1.5 with the input variable n
    //    set equal to 3.
    inverse_asymmetric_discrete_sine_transform_output_array_permutation<3>(data);

    // 9. Set T[ 1+2*i ] equal to -T[ 1+2*i ] for i = 0..3.
    for (auto i = 0u; i < 4; i++) {
        auto index = 1 + (2 * i);
        data[index] = -data[index];
    }
    return {};
}

template<typename T>
inline DecoderErrorOr<void> inverse_asymmetric_discrete_sine_transform_16(T* data)
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
        butterfly_rotation(data, high_precision_temp.data(), 2 * i, 1 + (2 * i), 31 - (4 * i), true);
    // 3. Invoke SH( i, 8+i ) for i = 0..7.
    for (auto i = 0u; i < 8; i++)
        hadamard_rotation(high_precision_temp.data(), data, i, 8 + i);

    // 4. Invoke SB( 8+2*i, 9+2*i, 28-16*i, 1 ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        butterfly_rotation(data, high_precision_temp.data(), 8 + (2 * i), 9 + (2 * i), 128 + 28 - (16 * i), true);
    // 5. Invoke SH( 8+i, 12+i ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        hadamard_rotation(high_precision_temp.data(), data, 8 + i, 12 + i);

    // 6. Invoke H( i, 4+i, 0 ) for i = 0..3.
    for (auto i = 0u; i < 4; i++)
        hadamard_rotation_in_place(data, i, 4 + i, false);

    // 7. Invoke SB( 4+8*i+3*j, 5+8*i+j, 24-16*j, 1 ) for i = 0..1, for j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            butterfly_rotation(data, high_precision_temp.data(), 4 + (8 * i) + (3 * j), 5 + (8 * i) + j, 24 - (16 * j), true);
    // 8. Invoke SH( 4+8*j+i, 6+8*j+i ) for i = 0..1, j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            hadamard_rotation(high_precision_temp.data(), data, 4 + (8 * j) + i, 6 + (8 * j) + i);

    // 9. Invoke H( 8*j+i, 2+8*j+i, 0 ) for i = 0..1, for j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            hadamard_rotation_in_place(data, (8 * j) + i, 2 + (8 * j) + i, false);
    // 10. Invoke B( 2+4*j+8*i, 3+4*j+8*i, 48+64*(i^j), 0 ) for i = 0..1, for j = 0..1.
    for (auto i = 0u; i < 2; i++)
        for (auto j = 0u; j < 2; j++)
            butterfly_rotation_in_place(data, 2 + (4 * j) + (8 * i), 3 + (4 * j) + (8 * i), 48 + (64 * (i ^ j)), false);

    // 11. Invoke the ADST output array permutation process specified in section 8.7.1.5 with the input variable n
    // set equal to 4.
    inverse_asymmetric_discrete_sine_transform_output_array_permutation<4>(data);

    // 12. Set T[ 1+12*j+2*i ] equal to -T[ 1+12*j+2*i ] for i = 0..1, for j = 0..1.
    for (auto i = 0u; i < 2; i++) {
        for (auto j = 0u; j < 2; j++) {
            auto index = 1 + (12 * j) + (2 * i);
            data[index] = -data[index];
        }
    }
    return {};
}

template<u8 log2_of_block_size, typename T>
inline DecoderErrorOr<void> inverse_asymmetric_discrete_sine_transform(T* data)
{
    // 8.7.1.9 Inverse ADST Process

    // This process performs an in-place inverse ADST process on the array T of size 2^n for 2 ≤ n ≤ 4.
    if constexpr (log2_of_block_size < 2 || log2_of_block_size > 4)
        return DecoderError::corrupted("Block size was out of range"sv);

    // The process to invoke depends on n as follows:
    if constexpr (log2_of_block_size == 2) {
        // − If n is equal to 2, invoke the Inverse ADST4 process specified in section 8.7.1.6.
        inverse_asymmetric_discrete_sine_transform_4(data);
        return {};
    }
    if constexpr (log2_of_block_size == 3) {
        // − Otherwise if n is equal to 3, invoke the Inverse ADST8 process specified in section 8.7.1.7.
        return inverse_asymmetric_discrete_sine_transform_8(data);
    }
    // − Otherwise (n is equal to 4), invoke the Inverse ADST16 process specified in section 8.7.1.8.
    return inverse_asymmetric_discrete_sine_transform_16(data);
}

}
