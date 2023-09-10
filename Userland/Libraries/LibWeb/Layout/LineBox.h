/*
 * Copyright (c) 2018-2020, Andreas Kling <kling@serenityos.org>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Vector.h>
#include <LibWeb/Layout/LineBoxFragment.h>

namespace Web::Layout {

class LineBox {
public:
    LineBox() = default;

    CSSPixelPoint position() const { return m_position; }
    CSSPixels width() const { return m_width; }
    CSSPixels height() const { return m_height; }
    CSSPixels bottom() const { return m_position.y() + m_height; }
    CSSPixels baseline_to_top() const { return m_baseline_to_top; }

    bool add_fragment(Node const& layout_node, size_t start, size_t length, CSSPixels leading_size, CSSPixels trailing_size, CSSPixels leading_margin, CSSPixels trailing_margin, CSSPixels content_width, CSSPixels content_height, CSSPixels border_box_top, CSSPixels border_box_bottom);

    Vector<LineBoxFragment> const& fragments() const { return m_fragments; }
    Vector<LineBoxFragment>& fragments() { return m_fragments; }

    void trim_trailing_whitespace();

    bool is_empty_or_ends_in_whitespace() const;
    bool is_empty() const { return m_fragments.is_empty() && !m_has_break; }

    AvailableSize available_width() const { return m_available_width; }

    CSSPixelRect const& absolute_rect() const { return m_absolute_rect; }
    void set_absolute_rect(CSSPixelRect const& rect) { m_absolute_rect = rect; }

private:
    friend class BlockContainer;
    friend class InlineFormattingContext;
    friend class LineBuilder;

    CSSPixelRect m_absolute_rect;

    Vector<LineBoxFragment> m_fragments;
    CSSPixelPoint m_position { 0, 0 };
    CSSPixels m_width { 0 };
    CSSPixels m_height { 0 };
    CSSPixels m_baseline_to_top { 0 };

    // The amount of available width that was originally available when creating this line box. Used for text justification.
    AvailableSize m_available_width { AvailableSize::make_indefinite() };

    bool m_has_break { false };
    bool m_has_forced_break { false };
};

}
