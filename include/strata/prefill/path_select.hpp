// path_select.hpp - which way a fresh prompt part is read: the decode windows, the Mac cold-expert lane (STRATA_PF_MAC) or the plain streaming prefill.
// Pure functions shared by generate.cpp (the windows/batched split), prefill.cpp (the lane / stream choice per chunk, the chunk size) and path_select_test.cpp.
//
// Why this exists (Prefill600, 2026-10-12): a chunk below the lane's minimum (default 256) did not take the lane; it fell back to the streaming prefill, which pushes
// every cold expert of every layer over the 190 MB/s USB link: 82-239 new tokens (just above --short-read 64) took 42-77 s (2-3 tok/s), while 26-60 tokens
// (windows) took 1-2.5 s and 368+ tokens (lane) 10-13 s.  The lane's cost is the SSD read of the cold experts, about the same for 70 or 2,000 tokens, so there is no
// size at which the streaming path wins: with the lane on, every batched chunk takes it.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

enum class ReadPath { Windows, MacLane, Stream };

// the largest chunk the lane's host buffers are sized for
inline constexpr int64_t kMacLaneMaxChunk = 16384;

// STRATA_PF_MAC_MIN default: the lane takes any chunk (the batched path is only entered for parts of more than --short-read tokens or picture rows)
inline constexpr int64_t kMacLaneMinDefault = 1;

// the path of one chunk of t tokens on the batched path
inline ReadPath batched_chunk_path(int64_t t, bool lane_on, int64_t lane_min) {
    return lane_on && t >= lane_min && t <= kMacLaneMaxChunk ? ReadPath::MacLane : ReadPath::Stream;
}

// STRATA_PF_EVEN: a lane prompt of more than one chunk of at most t_max tokens is cut into ceil(n / t_max) chunks of (nearly) equal size, rounded up to 16
inline int64_t even_chunk(int64_t n, int64_t t_max, bool even_on) {
    if (!even_on || n <= t_max) return t_max;
    const int64_t k = (n + t_max - 1) / t_max;
    return std::min<int64_t>(t_max, ((n + k - 1) / k + 15) & ~(int64_t) 15);
}

// the path of a fresh prompt part of n tokens: windows when n <= short_read, else the batched path, whose chunks (even_chunk) all take the same path unless a
// short tail falls below lane_min; returns the slowest path any chunk takes (Stream > MacLane > Windows)
inline ReadPath part_path(int64_t n, int64_t short_read, int64_t t_max, bool lane_on, int64_t lane_min, bool even_on = true) {
    if (n <= short_read) return ReadPath::Windows;
    const int64_t tc = lane_on ? even_chunk(n, t_max, even_on) : t_max;
    ReadPath worst = ReadPath::MacLane;
    for (int64_t c = 0; c < n; c += tc) {
        const int64_t t = std::min(tc, n - c);
        if (batched_chunk_path(t, lane_on, lane_min) == ReadPath::Stream) worst = ReadPath::Stream;
    }
    return worst;
}

}  // namespace strata::prefill
