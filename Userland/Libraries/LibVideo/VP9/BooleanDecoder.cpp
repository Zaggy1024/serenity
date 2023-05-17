/*
 * Copyright (c) 2021, Hunter Salyer <thefalsehonesty@gmail.com>
 * Copyright (c) 2022, Gregory Bertilson <zaggy1024@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/BuiltinWrappers.h>
#include <AK/Endian.h>

#include "BooleanDecoder.h"

namespace Video::VP9 {

// 9.2.1 Initialization process for Boolean decoder
ErrorOr<BooleanDecoder> BooleanDecoder::initialize(ReadonlyBytes data)
{
    if (data.size() < 1)
        return Error::from_string_literal("Size of decoder range cannot be zero");

    VERIFY(data.size() <= NumericLimits<i64>::max());
    // NOTE: As noted below in fill_reservoir(), we read in multi-byte-sized chunks,
    //       so here we will count the size in bytes rather than bits.
    BooleanDecoder decoder { data.data(), data.size() };

    if (decoder.read_bool(128))
        return Error::from_string_literal("Range decoder marker was non-zero");
    return decoder;
}

// Instead of filling the value field one bit at a time as the spec suggests,
// we store the value in more than 1 byte, filling those extra bytes all at once
// to avoid repeating logic and branching unnecessarily.
void BooleanDecoder::fill_reservoir()
{
    if (m_value_bits_left > 8)
        return;

    if (m_bytes_left == 0) {
        m_overread = true;
        return;
    }

    auto read_size = min<size_t>(reserve_bytes, m_bytes_left);
    ValueType read_value = 0;
    memcpy(&read_value, m_data, read_size);
    m_data += read_size;
    m_bytes_left -= read_size;

    read_value = AK::convert_between_host_and_big_endian(read_value);
    read_value >>= m_value_bits_left;
    m_value |= read_value;
    m_value_bits_left += read_size * 8;
}

// 9.2.2 Boolean decoding process
bool BooleanDecoder::read_bool(u8 probability)
{
    auto split = 1u + (((m_range - 1u) * probability) >> 8u);
    // The actual value being read resides in the most significant 8 bits
    // of the value field, so we shift the split into that range for comparison.
    auto split_shifted = static_cast<ValueType>(split) << reserve_bits;
    bool return_bool;

    if (m_value < split_shifted) {
        m_range = split;
        return_bool = false;
    } else {
        m_range -= split;
        m_value -= split_shifted;
        return_bool = true;
    }

    u8 bits_to_shift_into_range = count_leading_zeroes(m_range) - ((sizeof(m_range) - 1) * 8);
    m_range <<= bits_to_shift_into_range;
    m_value <<= bits_to_shift_into_range;
    m_value_bits_left -= bits_to_shift_into_range;

    fill_reservoir();

    return return_bool;
}

// 9.2.4 Parsing process for read_literal
u8 BooleanDecoder::read_literal(u8 bits)
{
    u8 return_value = 0;
    for (size_t i = 0; i < bits; i++) {
        return_value = (2 * return_value) + read_bool(128);
    }
    return return_value;
}

// 9.2.3 Exit process for Boolean decoder
ErrorOr<void> BooleanDecoder::finish_decode()
{
    if (m_overread)
        return Error::from_string_literal("Range decoder was read past the end of its data");

    bool padding_good = true;
    if (m_value != 0)
        padding_good = false;

    while (m_bytes_left > 0) {
        if (*m_data != 0)
            padding_good = false;
        m_data++;
        m_bytes_left--;
    }

    if (!padding_good)
        return Error::from_string_literal("Range decoder has a non-zero padding byte");

    // FIXME: It is a requirement of bitstream conformance that enough padding bits are inserted to ensure that the final coded byte of a frame is not equal to a superframe marker.
    //  A byte b is equal to a superframe marker if and only if (b & 0xe0)is equal to 0xc0, i.e. if the most significant 3 bits are equal to 0b110.
    return {};
}

}
