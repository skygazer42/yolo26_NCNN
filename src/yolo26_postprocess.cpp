#include "yolo26_postprocess.h"

#include <opencv2/core/core.hpp>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "ncnn/mat.h"

namespace {

float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409f * x + 126.93490512f);
    return v.f;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

float intersection_area(const Yolo26Object& a, const Yolo26Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent_inplace(std::vector<Yolo26Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(objects, left, j);
    if (i < right)
        qsort_descent_inplace(objects, i, right);
}

}  // namespace

namespace yolo26 {

void generate_proposals(const ncnn::Mat& reg, const ncnn::Mat& cls, int stride,
                        int input_w, int input_h, float prob_threshold,
                        std::vector<Yolo26Object>& objects)
{
    const int w = reg.w;
    const int h = reg.h;
    const int num_class = cls.c;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float max_score = -FLT_MAX;
            int label = -1;
            for (int c = 0; c < num_class; c++)
            {
                const float* cls_ptr = cls.channel(c);
                float score = cls_ptr[y * w + x];
                if (score > max_score)
                {
                    max_score = score;
                    label = c;
                }
            }

            float box_prob = sigmoid(max_score);
            if (box_prob < prob_threshold)
                continue;

            const float* reg_l = reg.channel(0);
            const float* reg_t = reg.channel(1);
            const float* reg_r = reg.channel(2);
            const float* reg_b = reg.channel(3);

            float grid_x = x + 0.5f;
            float grid_y = y + 0.5f;

            float x0 = (grid_x - reg_l[y * w + x]) * stride;
            float y0 = (grid_y - reg_t[y * w + x]) * stride;
            float x1 = (grid_x + reg_r[y * w + x]) * stride;
            float y1 = (grid_y + reg_b[y * w + x]) * stride;

            x0 = std::max(std::min(x0, (float)input_w - 1.f), 0.f);
            y0 = std::max(std::min(y0, (float)input_h - 1.f), 0.f);
            x1 = std::max(std::min(x1, (float)input_w - 1.f), 0.f);
            y1 = std::max(std::min(y1, (float)input_h - 1.f), 0.f);

            Yolo26Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            objects.push_back(obj);
        }
    }
}

void qsort_descent_inplace(std::vector<Yolo26Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, static_cast<int>(objects.size() - 1));
}

void nms_sorted_bboxes(const std::vector<Yolo26Object>& objects, std::vector<int>& picked,
                       float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Yolo26Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Yolo26Object& b = objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

}  // namespace yolo26
