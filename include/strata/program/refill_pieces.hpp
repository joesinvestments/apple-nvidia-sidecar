// include/strata/program/refill_pieces.hpp - the pieces of the guided expert refill (header-only: it is arithmetic).
//
// The refill copies a region's experts a piece at a time.  Piece k covers the slots from base + k * piece up to the
// next piece's first slot, and the LAST piece ends at the region's last copied slot, base + bound: never past it.
// `bound` is Kc, the slots copied before the first token.  Two things make it shorter than the region's slot count:
// the lazy refill (d18) leaves slots pending for the decode windows, and a region need not be the cache's last.  A
// clamp to the cache's slot count instead would copy slots the region has no expert for (assign[slot - base] reads
// past its own entries) and a piece could run past the staging buffer's span.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::program {

/// The pieces a region of `bound` slots is copied in (0 for 0 slots: nothing to copy before the first token).
inline int refill_pieces(std::size_t bound, int32_t piece) {
    piece = std::max<int32_t>(piece, 1);
    return (int) ((bound + (std::size_t) piece - 1) / (std::size_t) piece);
}

/// The slots [a, b) of piece `k` of the region that starts at slot `base` and copies `bound` of its slots.
/// A piece is at least one slot and at most `piece`; the pieces tile [base, base + bound) with no gap and no overlap.
inline void refill_piece(int32_t base, std::size_t bound, int32_t piece, int k, int32_t& a, int32_t& b) {
    piece = std::max<int32_t>(piece, 1);
    a = base + k * piece;
    b = std::min<int32_t>(base + (int32_t) bound, a + piece);
}

/// The slots of the widest piece: the staging buffer's span (its last slot's offset plus one blob).  The caller
/// sizes the buffer from this and every piece's soff(b - 1) stays inside it.
inline int32_t refill_span_slots(std::size_t bound, int32_t piece) {
    piece = std::max<int32_t>(piece, 1);
    return (int32_t) std::min<std::size_t>(bound, (std::size_t) piece);
}

}  // namespace strata::program
