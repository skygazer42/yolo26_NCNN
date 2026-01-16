#pragma once

#include <algorithm>
#include <vector>

namespace yolo26 {

struct TopKResult {
    float score = 0.f;
    int cls = -1;
    int anchor = -1;
};

template <typename ScoreGetter>
inline std::vector<TopKResult> get_topk_index(int anchors, int num_classes, int max_det, ScoreGetter get_score)
{
    std::vector<TopKResult> empty;
    if (anchors <= 0 || num_classes <= 0 || max_det <= 0)
        return empty;

    const int k = std::max(1, std::min(max_det, anchors));

    struct AnchorBest {
        float score;
        int anchor;
    };
    std::vector<AnchorBest> anchor_best;
    anchor_best.reserve((size_t)anchors);
    for (int a = 0; a < anchors; a++)
    {
        float best = get_score(a, 0);
        for (int c = 1; c < num_classes; c++)
            best = std::max(best, get_score(a, c));
        anchor_best.push_back({best, a});
    }

    std::partial_sort(anchor_best.begin(), anchor_best.begin() + k, anchor_best.end(),
                      [](const AnchorBest& lhs, const AnchorBest& rhs) {
                          if (lhs.score != rhs.score)
                              return lhs.score > rhs.score;
                          return lhs.anchor < rhs.anchor;
                      });
    anchor_best.resize(k);

    struct Candidate {
        float score;
        int cls;
        int anchor;
        int flat_index;
    };
    std::vector<Candidate> candidates;
    candidates.reserve((size_t)k * (size_t)num_classes);

    for (int p = 0; p < k; p++)
    {
        const int anchor = anchor_best[p].anchor;
        for (int c = 0; c < num_classes; c++)
        {
            candidates.push_back({get_score(anchor, c), c, anchor, p * num_classes + c});
        }
    }

    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                      [](const Candidate& lhs, const Candidate& rhs) {
                          if (lhs.score != rhs.score)
                              return lhs.score > rhs.score;
                          return lhs.flat_index < rhs.flat_index;
                      });
    candidates.resize(k);

    std::vector<TopKResult> results;
    results.reserve(candidates.size());
    for (const auto& cand : candidates)
    {
        TopKResult r;
        r.score = cand.score;
        r.cls = cand.cls;
        r.anchor = cand.anchor;
        results.push_back(r);
    }

    return results;
}

}  // namespace yolo26

