/*
 * Copyright (c) 2021, Hunter Salyer <thefalsehonesty@gmail.com>
 * Copyright (c) 2022, Gregory Bertilson <zaggy1024@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/ByteBuffer.h>
#include <AK/Error.h>
#include <AK/NonnullOwnPtr.h>
#include <AK/Queue.h>
#include <AK/Span.h>
#include <LibVideo/Color/CodingIndependentCodePoints.h>
#include <LibVideo/DecoderError.h>
#include <LibVideo/VideoDecoder.h>
#include <LibVideo/VideoFrame.h>

#include "Parser.h"

namespace Video::VP9 {

class Decoder : public VideoDecoder {
    friend class Parser;

public:
    Decoder();
    ~Decoder() override { }
    /* (8.1) General */
    DecoderErrorOr<void> receive_sample(ReadonlyBytes) override;

    DecoderErrorOr<NonnullOwnPtr<VideoFrame>> get_decoded_frame() override;

private:
    typedef i32 Intermediate;

    // Based on the maximum size resulting from num_4x4_blocks_wide_lookup.
    static constexpr size_t maximum_block_dimensions = 64ULL;
    static constexpr size_t maximum_block_size = maximum_block_dimensions * maximum_block_dimensions;
    // Based on the maximum for TXSize.
    static constexpr size_t maximum_transform_size = 32ULL * 32ULL;

    DecoderErrorOr<void> decode_frame(ReadonlyBytes);
    DecoderErrorOr<void> create_video_frame(FrameContext const&);

    DecoderErrorOr<void> allocate_buffers(FrameContext const&);
    Vector<u16>& get_output_buffer(u8 plane);

    /* (8.4) Probability Adaptation Process */
    u8 merge_prob(u8 pre_prob, u32 count_0, u32 count_1, u8 count_sat, u8 max_update_factor);
    u32 merge_probs(int const* tree, int index, u8* probs, u32* counts, u8 count_sat, u8 max_update_factor);
    DecoderErrorOr<void> adapt_coef_probs(FrameContext const&);
    DecoderErrorOr<void> adapt_non_coef_probs(FrameContext const&);
    void adapt_probs(int const* tree, u8* probs, u32* counts);
    u8 adapt_prob(u8 prob, u32 counts[2]);

    /* (8.5) Prediction Processes */
    // (8.5.1) Intra prediction process
    DecoderErrorOr<void> predict_intra(u8 plane, BlockContext const& block_context, u32 x, u32 y, bool have_left, bool have_above, bool not_on_right, TransformSize transform_size, u32 block_index);

    DecoderErrorOr<void> prepare_referenced_frame(Gfx::Size<u32> frame_size, u8 reference_frame_index);

    // (8.5.1) Inter prediction process
    DecoderErrorOr<void> predict_inter(u8 plane, BlockContext const& block_context, u32 x, u32 y, u32 width, u32 height, u32 block_index);
    // (8.5.2.1) Motion vector selection process
    MotionVector select_motion_vector(u8 plane, BlockContext const&, ReferenceIndex, u32 block_index);
    // (8.5.2.2) Motion vector clamping process
    MotionVector clamp_motion_vector(u8 plane, BlockContext const&, u32 block_row, u32 block_column, MotionVector vector);
    // From (8.5.1) Inter prediction process, steps 2-5
    DecoderErrorOr<void> predict_inter_block(u8 plane, BlockContext const&, ReferenceIndex, u32 block_row, u32 block_column, u32 x, u32 y, u32 width, u32 height, u32 block_index, u16* destination, u32 destination_stride);

    /* (8.6) Reconstruction and Dequantization */

    // Returns the quantizer index for the current block
    static u8 get_base_quantizer_index(SegmentFeatureStatus alternative_quantizer_feature, bool should_use_absolute_segment_base_quantizer, u8 base_quantizer_index);
    // Returns the quantizer value for the dc coefficient for a particular plane
    static u16 get_dc_quantizer(u8 bit_depth, u8 base, i8 delta);
    // Returns the quantizer value for the ac coefficient for a particular plane
    static u16 get_ac_quantizer(u8 bit_depth, u8 base, i8 delta);

    // (8.6.2) Reconstruct process
    DecoderErrorOr<void> reconstruct(u8 plane, BlockContext const&, u32 transform_block_x, u32 transform_block_y, TransformSize transform_block_size, TransformSet);
    template<u8 log2_of_block_size>
    DecoderErrorOr<void> reconstruct_templated(u8 plane, BlockContext const&, u32 transform_block_x, u32 transform_block_y, TransformSet);

    // (8.7) Inverse transform process
    template<u8 log2_of_block_size>
    DecoderErrorOr<void> inverse_transform_2d(BlockContext const&, Span<Intermediate> dequantized, TransformSet);
    template<u8 log2_of_block_size, TransformSet>
    DecoderErrorOr<void> inverse_transform_2d_templated(BlockContext const&, Span<Intermediate> dequantized);

    /* (8.10) Reference Frame Update Process */
    DecoderErrorOr<void> update_reference_frames(FrameContext const&);

    NonnullOwnPtr<Parser> m_parser;

    Vector<u16> m_output_buffers[3];

    Queue<NonnullOwnPtr<VideoFrame>, 1> m_video_frame_queue;
};

}
