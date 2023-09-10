/*
 * Copyright (c) 2022, Andreas Kling <kling@serenityos.org>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <LibWeb/Layout/BlockFormattingContext.h>
#include <LibWeb/Layout/LineBuilder.h>
#include <LibWeb/Layout/TextNode.h>

namespace Web::Layout {

LineBuilder::LineBuilder(InlineFormattingContext& context, LayoutState& layout_state)
    : m_context(context)
    , m_layout_state(layout_state)
    , m_containing_block_state(layout_state.get_mutable(context.containing_block()))
{
    m_text_indent = m_context.containing_block().computed_values().text_indent().to_px(m_context.containing_block(), m_containing_block_state.content_width());
    begin_new_line();
}

LineBuilder::~LineBuilder()
{
    if (m_last_line_needs_update)
        update_last_line();
}

void LineBuilder::break_line(ForcedBreak forced_break, Optional<CSSPixels> next_item_width)
{
    auto& previous_line_box = last_line_box();
    previous_line_box.m_has_break = true;
    previous_line_box.m_has_forced_break = forced_break == ForcedBreak::Yes;

    update_last_line();

    m_current_y += previous_line_box.height();

    while (true) {
        recalculate_available_space();
        auto floats_intrude_at_current_y = m_context.any_floats_intrude_at_y(m_current_y);
        if (!floats_intrude_at_current_y)
            break;
        if (m_context.can_fit_new_line_at_y(m_current_y)) {
            if (!next_item_width.has_value() || !(next_item_width.value() > m_available_width_for_current_line))
                break;
        }
        m_current_y += 1;
    }
    begin_new_line();
}

// https://drafts.csswg.org/css2/#leading
class InlineMetrics {
public:
    // From https://drafts.csswg.org/css2/#propdef-vertical-align:
    // In the following definitions, for inline non-replaced elements, the box used for alignment is
    // the box whose height is the line-height (containing the box’s glyphs and the half-leading on
    // each side, see above). For all other elements, the box used for alignment is the margin box. 

    static InlineMetrics from_inline_box(Node const& node)
    {
        InlineMetrics metrics;
        metrics.m_line_height = node.line_height();

        auto const& font_metrics = node.font().pixel_metrics();
        auto combined = CSSPixels::nearest_value_for(font_metrics.ascent + font_metrics.descent);
        metrics.m_ascent = CSSPixels::nearest_value_for(font_metrics.ascent);
        // Ensure that our conversion to CSSPixels has a total as close to the actual A+D as possible.
        metrics.m_descent = combined - metrics.m_ascent;
        metrics.m_x_height = CSSPixels::nearest_value_for(font_metrics.x_height);

        return metrics;
    }

    static InlineMetrics from_atomic_inline(FormattingContext const& context, Box const& box)
    {
        InlineMetrics metrics;

        metrics.m_line_height = context.margin_box_rect(box).height();
        metrics.m_ascent = context.box_baseline(box);
        metrics.m_x_height = metrics.m_ascent;
        metrics.m_descent = metrics.m_line_height - metrics.m_ascent;

        return metrics;
    }

    CSSPixels ascent() const
    {
        return m_ascent;
    }
    CSSPixels descent() const
    {
        return m_descent;
    }
    CSSPixels x_height() const
    {
        return m_x_height;
    }
    CSSPixels line_height() const
    {
        return m_line_height;
    }

    CSSPixels leading() const
    {
        return max(m_line_height - m_ascent - m_descent, 0);
    }
    CSSPixels leading_top() const
    {
        return leading() / 2;
    }
    CSSPixels leading_bottom() const
    {
        return leading() - leading_top();
    }
    CSSPixels top() const
    {
        return m_ascent + leading_top();
    }
    CSSPixels bottom() const
    {
        return m_descent + leading_bottom();
    }

private:
    InlineMetrics() = default;

    CSSPixels m_ascent { 0 };
    CSSPixels m_descent { 0 };
    CSSPixels m_x_height { 0 };
    CSSPixels m_line_height { 0 };
};

void LineBuilder::begin_new_line()
{
    LineBox line_box;
    // On a block container element whose content is composed of inline-level elements, 'line-height' specifies
    // the minimal height of line boxes within the element. The minimum height consists of a minimum height above
    // the baseline and a minimum depth below it, exactly as if each line box starts with a zero-width inline box
    // with the element’s font and line height properties. We call that imaginary box a "strut."
    auto strut_metrics = InlineMetrics::from_inline_box(m_context.containing_block());
    m_current_line_baseline_to_top = strut_metrics.top();
    m_current_line_baseline_to_bottom = strut_metrics.bottom();
    line_box.m_height = m_current_line_baseline_to_top + m_current_line_baseline_to_bottom;

    // FIXME: Support text-indent with "each-line".
    bool indented = m_containing_block_state.line_boxes.is_empty();
    line_box.m_position = { indented ? m_text_indent : 0, m_current_y };

    m_containing_block_state.line_boxes.append(move(line_box));

    recalculate_available_space();
    last_line_box().m_available_width = m_available_width_for_current_line;
    m_last_line_needs_update = true;
}

LineBox& LineBuilder::last_line_box()
{
    return m_containing_block_state.line_boxes.last();
}

void LineBuilder::append_box(Box const& box, CSSPixels leading_size, CSSPixels trailing_size, CSSPixels leading_margin, CSSPixels trailing_margin)
{
    auto& box_state = m_layout_state.get_mutable(box);
    auto& line_box = last_line_box();
    if (line_box.add_fragment(box, 0, 0, leading_size, trailing_size, leading_margin, trailing_margin, box_state.content_width(), box_state.content_height(), box_state.border_box_top(), box_state.border_box_bottom()))
        after_fragment_appended();

    box_state.containing_line_box_fragment = LineBoxFragmentCoordinate {
        .line_box_index = m_containing_block_state.line_boxes.size() - 1,
        .fragment_index = line_box.fragments().size() - 1,
    };
}

void LineBuilder::append_text_chunk(TextNode const& text_node, size_t offset_in_node, size_t length_in_node, CSSPixels leading_size, CSSPixels trailing_size, CSSPixels leading_margin, CSSPixels trailing_margin, CSSPixels content_width, CSSPixels content_height)
{
    if (last_line_box().add_fragment(text_node, offset_in_node, length_in_node, leading_size, trailing_size, leading_margin, trailing_margin, content_width, content_height, 0, 0))
        after_fragment_appended();
}

void LineBuilder::after_fragment_appended()
{
    // As described in the section on inline formatting contexts, user agents flow inline-level boxes
    // into a vertical stack of line boxes. The height of a line box is determined as follows:
    auto& line_box = last_line_box();

    auto& fragment = line_box.fragments().last();
    VERIFY(fragment.layout_node().containing_block() != nullptr);

    // 10.8. Line height calculations: the 'line-height' and 'vertical-align' properties
    // 1. The height of each inline-level box in the line box is calculated.
    //    For replaced elements, inline-block elements, and inline-table elements, this is the height of their margin box;
    //    for inline boxes, this is their line-height. (See "Calculating heights and margins" and the height of inline boxes in "Leading and half-leading".)

    auto fragment_metrics = [&]() {
        if (fragment.is_atomic_inline()) {
            auto const& fragment_box = static_cast<Box const&>(fragment.layout_node());
            auto metrics = InlineMetrics::from_atomic_inline(m_context, fragment_box);
            // According to the 'baseline' value of 'vertical-align', we should treat the bottom of the margin box as the
            // baseline here: "If the box does not have a baseline, align the bottom margin edge with the parent’s baseline."
            // Painting coordinates for boxes are at the top left of the content box, so we need to offset upwards by the
            // distance from the margin bottom to the content top.
            auto const& layout_state = m_layout_state.get(fragment_box);
            fragment.set_offset(fragment.offset().translated(0, metrics.bottom() - (layout_state.content_height() + layout_state.margin_box_bottom())));
            return metrics;
        }

        auto metrics = InlineMetrics::from_inline_box(fragment.layout_node());;
        // We want to place the fragment's bounding box at the top of the font, so we take away the distance to the top here.
        // The text baseline offset will correct this when rendering text which expects the painting coordinate to be at the
        // baseline of the text.
        fragment.set_text_baseline_offset(metrics.top());
        fragment.set_offset(fragment.offset().translated(0, -metrics.top()));
        return metrics;
    }();

    // 2. The inline-level boxes are aligned vertically according to their vertical-align property.
    auto offset_above_baseline = [&]() -> Optional<CSSPixels> {
        auto const& vertical_align = fragment.layout_node().computed_values().vertical_align();
        if (vertical_align.has<CSS::VerticalAlign>()) {
            auto parent_metrics = InlineMetrics::from_inline_box(m_context.containing_block());

            switch (vertical_align.get<CSS::VerticalAlign>()) {
            case CSS::VerticalAlign::Baseline:
                // Align the baseline of the box with the baseline of the parent box.
                // If the box does not have a baseline, align the bottom margin edge with the parent’s baseline.
                return 0;

            // In case they are aligned top or bottom, they must be aligned so as to minimize the line box height.
            // If such boxes are tall enough, there are multiple solutions and CSS 2 does not define
            // the position of the line box’s baseline (i.e., the position of the strut, see below).

            // To accomplish this, we can skip affecting the top and bottom spacing of the spacing from the line box's
            // baseline to its top and bottom. We only change the line box's effective height instead, since we must
            // always have enough room for the fragment, but if the line box would otherwise remain smaller than the
            // top/bottom-aligned box, we want it to become exactly the size of that box.
            //
            // Then, when we have completely determined the baseline of all the other fragments in the line box, we
            // can place the top/bottom-aligned boxes vertically, extending either the bottom or top distance of the
            // baseline to accomodate the aligned elements. See update_last_line().
            case CSS::VerticalAlign::Top:
            case CSS::VerticalAlign::Bottom:
                line_box.m_height = max(fragment_metrics.line_height(), line_box.m_height);
                return {};

            case CSS::VerticalAlign::Middle: {
                // Align the vertical midpoint of the box with the baseline of the parent box plus half the x-height of the parent.
                auto baseline_to_box_middle = (fragment_metrics.ascent() + fragment_metrics.descent()) / 2 - fragment_metrics.descent();
                return parent_metrics.x_height() / 2 - baseline_to_box_middle;
            }
            case CSS::VerticalAlign::Sub:
                // FIXME: Lower the baseline of the box to the proper position for subscripts of the parent’s box.
            case CSS::VerticalAlign::Super:
                // FIXME: Raise the baseline of the box to the proper position for superscripts of the parent’s box.
                return 0;
            case CSS::VerticalAlign::TextTop:
                // Align the top of the box with the top of the parent’s content area (see 10.6.1).
                // NOTE: 10.6.1. indicates that the content area should be based on the font, but does not specify
                //       exactly how. One suggested method is to use the ascender and descender, so let's align
                //       to those, since we conveniently have them here.
                return parent_metrics.ascent() - fragment_metrics.top();
            case CSS::VerticalAlign::TextBottom:
                // Align the bottom of the box with the bottom of the parent’s content area (see 10.6.1).
                return fragment_metrics.bottom() - parent_metrics.descent();
            }
            VERIFY_NOT_REACHED();
        }

        auto const& length_percentage = vertical_align.get<CSS::LengthPercentage>();
        if (length_percentage.is_percentage()) {
            // Raise (positive value) or lower (negative value) the box by this distance (a percentage of the line-height value).
            // The value 0% means the same as baseline.
            auto vertical_align_amount = m_context.containing_block().line_height().scaled(length_percentage.percentage().as_fraction());
            return vertical_align_amount;
        }
        VERIFY(length_percentage.is_length());
        // Raise (positive value) or lower (negative value) the box by this distance.
        // The value 0cm means the same as baseline.
        auto vertical_align_amount = length_percentage.length().to_px(fragment.layout_node());
        return vertical_align_amount;
    }();

    if (offset_above_baseline.has_value()) {
        // 3. The line box height is the distance between the uppermost box top and the lowermost box bottom.
        m_current_line_baseline_to_top = max(fragment_metrics.top() + offset_above_baseline.value(), m_current_line_baseline_to_top);
        m_current_line_baseline_to_bottom = max(fragment_metrics.bottom() - offset_above_baseline.value(), m_current_line_baseline_to_bottom);

        fragment.set_offset(fragment.offset().translated(0, -offset_above_baseline.value()));
    }

    line_box.m_height = max(m_current_line_baseline_to_top + m_current_line_baseline_to_bottom, line_box.m_height);
    line_box.m_baseline_to_top = max(m_current_line_baseline_to_top, line_box.m_baseline_to_top);
}

CSSPixels LineBuilder::y_for_float_to_be_inserted_here(Box const& box)
{
    auto const& box_state = m_layout_state.get(box);
    CSSPixels const width = box_state.margin_box_width();
    CSSPixels const height = box_state.margin_box_height();

    CSSPixels candidate_y = m_current_y;

    CSSPixels current_line_width = last_line_box().width();
    // If there's already inline content on the current line, check if the new float can fit
    // alongside the content. If not, place it on the next line.
    if (current_line_width > 0 && (current_line_width + width) > m_available_width_for_current_line)
        candidate_y += m_context.containing_block().line_height();

    // Then, look for the next Y position where we can fit the new float.
    // FIXME: This is super dumb, we move 1px downwards per iteration and stop
    //        when we find an Y value where we don't collide with other floats.
    while (true) {
        auto space_at_y_top = m_context.available_space_for_line(candidate_y);
        auto space_at_y_bottom = m_context.available_space_for_line(candidate_y + height);
        if (width > space_at_y_top || width > space_at_y_bottom) {
            if (!m_context.any_floats_intrude_at_y(candidate_y) && !m_context.any_floats_intrude_at_y(candidate_y + height)) {
                return candidate_y;
            }
        } else {
            return candidate_y;
        }
        candidate_y += 1;
    }
}

bool LineBuilder::should_break(CSSPixels next_item_width)
{
    if (m_available_width_for_current_line.is_max_content())
        return false;

    auto const& line_boxes = m_containing_block_state.line_boxes;
    if (line_boxes.is_empty() || line_boxes.last().is_empty()) {
        // If we don't have a single line box yet *and* there are no floats intruding
        // at this Y coordinate, we don't need to break before inserting anything.
        if (!m_context.any_floats_intrude_at_y(m_current_y))
            return false;
        if (!m_context.any_floats_intrude_at_y(m_current_y + m_context.containing_block().line_height()))
            return false;
    }
    auto current_line_width = last_line_box().width();
    return (current_line_width + next_item_width) > m_available_width_for_current_line;
}

void LineBuilder::update_last_line()
{
    VERIFY(m_last_line_needs_update);
    m_last_line_needs_update = false;
    auto& line_box = last_line_box();

    // Calculate the horizontal alignment offset.
    auto text_align = m_context.containing_block().computed_values().text_align();

    CSSPixels x_offset_top = m_context.leftmost_x_offset_at(m_current_y);
    CSSPixels x_offset_bottom = m_context.leftmost_x_offset_at(m_current_y + line_box.height() - 1);
    CSSPixels x_offset = max(x_offset_top, x_offset_bottom);

    CSSPixels excess_horizontal_space = m_available_width_for_current_line.to_px_or_zero() - line_box.width();

    // If (after justification, if any) the inline contents of a line box are too long to fit within it,
    // then the contents are start-aligned: any content that doesn't fit overflows the line box’s end edge.
    if (excess_horizontal_space > 0) {
        switch (text_align) {
        case CSS::TextAlign::Center:
        case CSS::TextAlign::LibwebCenter:
            x_offset += excess_horizontal_space / 2;
            break;
        case CSS::TextAlign::Right:
        case CSS::TextAlign::LibwebRight:
            x_offset += excess_horizontal_space;
            break;
        case CSS::TextAlign::Left:
        case CSS::TextAlign::LibwebLeft:
        case CSS::TextAlign::Justify:
        default:
            break;
        }
    }

    // Run through top/bottom-aligned boxes. If there is room for a box, we should only place the box,
    // and not affect the line box size. If there is not enough room, we will expand the line box in
    // the opposite direction of its alignment. This is not defined by the spec, but matches the behavior
    // of Blink and Chrome.
    for (auto& fragment : last_line_box().fragments()) {
        auto fragment_metrics = [&]() {
            if (fragment.is_atomic_inline())
                return InlineMetrics::from_atomic_inline(m_context, static_cast<Box const&>(fragment.layout_node()));

            auto metrics = InlineMetrics::from_inline_box(fragment.layout_node());
            return metrics;
        }();

        auto const& vertical_align = fragment.layout_node().computed_values().vertical_align();
        if (vertical_align.has<CSS::VerticalAlign>()) {
            switch (vertical_align.get<CSS::VerticalAlign>()) {
            case CSS::VerticalAlign::Top:
                // Align the top of the aligned subtree with the top of the line box.
                m_current_line_baseline_to_bottom = max(m_current_line_baseline_to_bottom, fragment_metrics.line_height() - m_current_line_baseline_to_top);
                fragment.set_offset(fragment.offset().translated(0, fragment_metrics.top() - m_current_line_baseline_to_top));
                break;
            case CSS::VerticalAlign::Bottom:
                // Align the top of the aligned subtree with the top of the line box.
                m_current_line_baseline_to_top = max(m_current_line_baseline_to_top, fragment_metrics.line_height() - m_current_line_baseline_to_bottom);
                fragment.set_offset(fragment.offset().translated(0, m_current_line_baseline_to_bottom - fragment_metrics.bottom()));
                break;
            default:
                break;
            }
        }
    }

    // At this point, we have all fragments aligned so that 0 is the baseline.
    // Offset all line box fragments according to the alignment of the line box,
    // and apply the current y offset.
    auto current_line_box_height = m_current_line_baseline_to_top + m_current_line_baseline_to_bottom;
    line_box.m_height = current_line_box_height;
    line_box.m_baseline_to_top = m_current_line_baseline_to_top;
    for (auto& fragment : line_box.fragments())
        fragment.set_offset({ fragment.offset().x() + x_offset, fragment.offset().y() + m_current_y + m_current_line_baseline_to_top });
}

void LineBuilder::remove_last_line_if_empty()
{
    // If there's an empty line box at the bottom, just remove it instead of giving it height.
    auto& line_boxes = m_containing_block_state.line_boxes;
    if (!line_boxes.is_empty() && line_boxes.last().is_empty()) {
        line_boxes.take_last();
        m_last_line_needs_update = false;
    }
}

void LineBuilder::recalculate_available_space()
{
    auto current_line_height = last_line_box().height();
    auto available_at_top_of_line_box = m_context.available_space_for_line(m_current_y);
    auto available_at_bottom_of_line_box = m_context.available_space_for_line(m_current_y + current_line_height - 1);
    m_available_width_for_current_line = min(available_at_bottom_of_line_box, available_at_top_of_line_box);
    last_line_box().m_available_width = m_available_width_for_current_line;
}

}
