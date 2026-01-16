#pragma once

#include <algorithm>
#include <vector>

namespace yolo26 {

inline float compute_iou(float x1a, float y1a, float x2a, float y2a,
                         float x1b, float y1b, float x2b, float y2b)
{
    float inter_x1 = std::max(x1a, x1b);
    float inter_y1 = std::max(y1a, y1b);
    float inter_x2 = std::min(x2a, x2b);
    float inter_y2 = std::min(y2a, y2b);

    float inter_w = std::max(0.f, inter_x2 - inter_x1);
    float inter_h = std::max(0.f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (x2a - x1a) * (y2a - y1a);
    float area_b = (x2b - x1b) * (y2b - y1b);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.f)
        return 0.f;
    return inter_area / union_area;
}

template <typename Object>
inline std::vector<Object> nms(const std::vector<Object>& objects, float iou_threshold, bool agnostic = false)
{
    if (objects.empty())
        return objects;

    std::vector<Object> sorted_objs = objects;
    std::sort(sorted_objs.begin(), sorted_objs.end(),
              [](const Object& a, const Object& b) { return a.prob > b.prob; });

    std::vector<bool> suppressed(sorted_objs.size(), false);
    std::vector<Object> result;
    result.reserve(sorted_objs.size());

    for (size_t i = 0; i < sorted_objs.size(); i++)
    {
        if (suppressed[i])
            continue;

        result.push_back(sorted_objs[i]);
        const Object& obj_i = sorted_objs[i];

        for (size_t j = i + 1; j < sorted_objs.size(); j++)
        {
            if (suppressed[j])
                continue;

            const Object& obj_j = sorted_objs[j];

            if (!agnostic && obj_i.label != obj_j.label)
                continue;

            float iou = compute_iou(obj_i.x1, obj_i.y1, obj_i.x2, obj_i.y2,
                                    obj_j.x1, obj_j.y1, obj_j.x2, obj_j.y2);
            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }

    return result;
}

}  // namespace yolo26
