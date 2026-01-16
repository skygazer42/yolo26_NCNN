#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core/core.hpp>

#include "mat.h"

#include "yolo26_resize.h"

namespace yolo26 {

struct BoxXYXY {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
};

inline int round_to_even(float v)
{
    return (int)std::nearbyint((double)v);
}

inline void crop_mask_inplace(float* mask, int h, int w, const BoxXYXY& box)
{
    if (!mask || h <= 0 || w <= 0)
        return;

    int x1 = std::max(0, std::min(round_to_even(box.x1), w));
    int y1 = std::max(0, std::min(round_to_even(box.y1), h));
    int x2 = std::max(0, std::min(round_to_even(box.x2), w));
    int y2 = std::max(0, std::min(round_to_even(box.y2), h));

    if (x2 <= x1 || y2 <= y1)
    {
        std::fill(mask, mask + (size_t)h * (size_t)w, 0.f);
        return;
    }

    for (int y = 0; y < y1; y++)
        std::fill(mask + (size_t)y * (size_t)w, mask + (size_t)(y + 1) * (size_t)w, 0.f);
    for (int y = y2; y < h; y++)
        std::fill(mask + (size_t)y * (size_t)w, mask + (size_t)(y + 1) * (size_t)w, 0.f);

    for (int y = y1; y < y2; y++)
    {
        float* row = mask + (size_t)y * (size_t)w;
        if (x1 > 0)
            std::fill(row, row + x1, 0.f);
        if (x2 < w)
            std::fill(row + x2, row + w, 0.f);
    }
}

inline bool process_mask(const ncnn::Mat& protos,
                         const ncnn::Mat& masks_in,
                         const std::vector<BoxXYXY>& bboxes_xyxy,
                         int shape_h,
                         int shape_w,
                         bool upsample,
                         std::vector<cv::Mat>& out_masks)
{
    out_masks.clear();
    if (protos.dims != 3 || masks_in.dims != 2 || shape_h <= 0 || shape_w <= 0)
        return false;

    const int c = protos.c;
    const int mh = protos.h;
    const int mw = protos.w;
    const int n = masks_in.h;
    if (c <= 0 || mh <= 0 || mw <= 0 || n <= 0)
        return false;
    if (masks_in.w != c)
        return false;
    if ((int)bboxes_xyxy.size() != n)
        return false;

    cv::Mat coeffs(n, c, CV_32FC1);
    for (int i = 0; i < n; i++)
    {
        const float* src = masks_in.row(i);
        float* dst = coeffs.ptr<float>(i);
        std::copy(src, src + c, dst);
    }

    cv::Mat protos_flat(c, mw * mh, CV_32FC1, (void*)protos.data);
    cv::Mat mask_logits;
    cv::gemm(coeffs, protos_flat, 1.0, cv::Mat(), 0.0, mask_logits);  // (n, mw*mh)

    const float width_ratio = mw / (float)shape_w;
    const float height_ratio = mh / (float)shape_h;

    out_masks.reserve(n);
    for (int i = 0; i < n; i++)
    {
        cv::Mat m = mask_logits.row(i).reshape(1, mh);  // (mh, mw) logits
        BoxXYXY box = bboxes_xyxy[i];
        box.x1 *= width_ratio;
        box.x2 *= width_ratio;
        box.y1 *= height_ratio;
        box.y2 *= height_ratio;

        crop_mask_inplace((float*)m.data, mh, mw, box);

        cv::Mat mask_src = m;
        if (upsample && (mh != shape_h || mw != shape_w))
        {
            cv::Mat up(shape_h, shape_w, CV_32FC1);
            resize_bilinear_align_false((const float*)mask_src.data, mh, mw, (float*)up.data, shape_h, shape_w);
            mask_src = up;
        }

        cv::Mat bin(mask_src.rows, mask_src.cols, CV_8UC1);
        const float* mp = (const float*)mask_src.data;
        unsigned char* bp = bin.data;
        const int total = mask_src.rows * mask_src.cols;
        for (int j = 0; j < total; j++)
            bp[j] = (mp[j] > 0.f) ? 1 : 0;
        out_masks.push_back(std::move(bin));
    }

    return true;
}

}  // namespace yolo26

