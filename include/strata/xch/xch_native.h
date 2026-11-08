// include/strata/xch/xch_native.h - the host reference of the cold-tolerant resident plan, for the exchange
// window of an all-NVFP4 native pack.
//
// Under the exchange (set_xch) the card computes the layer's RESIDENT experts from its VRAM cache and Metal computes
// the cold ones; the card's plan is built on the device by kernels::resident_plan with `allow_cold`, which skips a
// cold id instead of emptying the plan (#871's strict behaviour stays the default).  This header states that plan on
// the host, so the kernel and a test can be held to one rule.
#pragma once

#include <vector>

/// One group's cold-tolerant resident plan: `experts[g]` is routed by `counts[g]` of the group's entries, whose global
/// pair indices (row * k + j, ascending) are the next `counts[g]` of `dst`.  `err`: a cold id with allow_cold false.
struct XchPlanRef {
    std::vector<int> experts, counts, dst;
    bool err = false;
};

/// ids: a group's routed ids (n = rows x k entries).  res[e] = the VRAM slot of expert e, or -1 when it is cold.
/// Groups are the distinct RESIDENT experts in order of first appearance; dst lists, group by group, the entries'
/// global pair indices ascending.  A cold id with allow_cold false empties the plan and sets err (today's #871
/// behaviour); with allow_cold true it is skipped and err stays false.
inline XchPlanRef xch_resident_plan_ref(const std::vector<int>& ids, const std::vector<int>& res, bool allow_cold) {
    XchPlanRef p;
    const int n = (int) ids.size();
    auto resident = [&](int i) {
        const int e = ids[(size_t) i];
        return e >= 0 && e < (int) res.size() && res[(size_t) e] >= 0;
    };
    for (int i = 0; i < n; ++i)
        if (!resident(i) && !allow_cold) {   // #871: a cold pair and no tolerance: an empty plan
            p.err = true;
            return p;
        }
    std::vector<int> group_of((size_t) n, -1);
    for (int i = 0; i < n; ++i) {
        if (!resident(i)) continue;   // a cold entry is not a group member
        const int e = ids[(size_t) i];
        int g = -1;
        for (size_t j = 0; j < p.experts.size(); ++j)
            if (p.experts[j] == e) { g = (int) j; break; }
        if (g < 0) {
            g = (int) p.experts.size();
            p.experts.push_back(e);
            p.counts.push_back(0);
        }
        group_of[(size_t) i] = g;
        ++p.counts[(size_t) g];
    }
    for (size_t g = 0; g < p.experts.size(); ++g)
        for (int i = 0; i < n; ++i)
            if (group_of[(size_t) i] == (int) g) p.dst.push_back(i);
    return p;
}
