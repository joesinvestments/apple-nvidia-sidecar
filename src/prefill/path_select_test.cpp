// path_select_test.cpp - card-free test of the read-path selection (include/strata/prefill/path_select.hpp).
// Build: clang++ -std=c++20 -I include src/prefill/path_select_test.cpp -o path_select_test && ./path_select_test
#include <cstdio>
#include <cstdlib>

#include "strata/prefill/path_select.hpp"

using namespace strata::prefill;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    const int64_t T = 2048, SHORT = 64;

    // the anomaly: with the lane's old minimum (256) every part of 65..255 tokens streamed its experts over the link (42-77 s on the card)
    for (int64_t n = SHORT + 1; n < 256; ++n) CHECK(part_path(n, SHORT, T, true, 256) == ReadPath::Stream);
    for (int64_t n = 256; n <= 3000; ++n) CHECK(part_path(n, SHORT, T, true, 256) == ReadPath::MacLane || n > 2048);   // (n > 2048: a tail below 256 can still stream)
    CHECK(part_path(2048 + 100, SHORT, T, true, 256) == ReadPath::MacLane);   // even chunks 1074 + 1074: no tail pass
    CHECK(part_path(2048 * 3 + 5, SHORT, T, true, 256) == ReadPath::MacLane);

    // the fix: with the new default no part ever takes the streaming path while the lane is on
    for (int64_t n = 1; n <= 40000; ++n) {
        const ReadPath p = part_path(n, SHORT, T, true, kMacLaneMinDefault);
        if (n <= SHORT) CHECK(p == ReadPath::Windows);
        else CHECK(p == ReadPath::MacLane);
    }
    // picture rows / other callers enter the batched path with any size, including 1..64 tokens
    for (int64_t t = 1; t <= T; ++t) CHECK(batched_chunk_path(t, true, kMacLaneMinDefault) == ReadPath::MacLane);

    // without the lane the batched path is the streaming one, whatever the size
    for (int64_t n = SHORT + 1; n <= 5000; n += 7) CHECK(part_path(n, SHORT, T, false, kMacLaneMinDefault) == ReadPath::Stream);

    // chunking: equal chunks <= T, a multiple of 16, covering the part; never a runt tail
    for (int64_t n = 1; n <= 70000; n += 13) {
        const int64_t tc = even_chunk(n, T, true);
        CHECK(tc >= 1 && tc <= T);
        if (n > T) {
            CHECK(tc % 16 == 0);
            const int64_t k = (n + tc - 1) / tc;
            CHECK(k == (n + T - 1) / T);              // as many chunks as the plain split
            CHECK(n - (k - 1) * tc >= tc - 16 * (k + 1));   // the last chunk is within the rounding of the others
        } else CHECK(tc == T);
    }
    CHECK(even_chunk(8218, T, true) == 1648);          // the measured case: 5 x 1648, not 4 x 2048 + 26
    CHECK(even_chunk(8218, T, false) == T);

    if (fails) { std::printf("%d failure(s)\n", fails); return 1; }
    std::printf("path_select_test: all ok\n");
    return 0;
}
