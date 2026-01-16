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

inline bool scale_mask(const cv::Mat& in,
                       int target_h,
                       int target_w,
                       cv::Mat& out,
                       bool padding = true)
{
    out.release();
    if (in.empty() || in.type() != CV_32FC1 || target_h <= 0 || target_w <= 0)
        return false;

    const int in_h = in.rows;
    const int in_w = in.cols;
    if (in_h == target_h && in_w == target_w)
    {
        out = in.clone();
        return true;
    }

    const float gain = std::min(in_h / (float)target_h, in_w / (float)target_w);
    float pad_w = in_w - target_w * gain;
    float pad_h = in_h - target_h * gain;
    if (padding)
    {
        pad_w /= 2.f;
        pad_h /= 2.f;
    }

    const int top = padding ? (int)std::round(pad_h - 0.1f) : 0;
    const int left = padding ? (int)std::round(pad_w - 0.1f) : 0;
    const int bottom = in_h - (int)std::round(pad_h + 0.1f);
    const int right = in_w - (int)std::round(pad_w + 0.1f);

    const int crop_x = std::max(0, std::min(left, in_w));
    const int crop_y = std::max(0, std::min(top, in_h));
    const int crop_w = std::max(0, std::min(right, in_w) - crop_x);
    const int crop_h = std::max(0, std::min(bottom, in_h) - crop_y);
    if (crop_w <= 0 || crop_h <= 0)
        return false;

    cv::Mat roi = in(cv::Rect(crop_x, crop_y, crop_w, crop_h));
    std::vector<float> roi_buf((size_t)crop_h * (size_t)crop_w);
    for (int y = 0; y < crop_h; y++)
    {
        const float* src = roi.ptr<float>(y);
        float* dst = roi_buf.data() + (size_t)y * (size_t)crop_w;
        std::copy(src, src + crop_w, dst);
    }

    out = cv::Mat(target_h, target_w, CV_32FC1);
    resize_bilinear_align_false(roi_buf.data(), crop_h, crop_w, out.ptr<float>(), target_h, target_w);
    return true;
}

inline bool scale_masks(const std::vector<cv::Mat>& in_masks,
                        int target_h,
                        int target_w,
                        std::vector<cv::Mat>& out_masks,
                        bool padding = true)
{
    out_masks.clear();
    out_masks.reserve(in_masks.size());
    for (const auto& mask : in_masks)
    {
        if (mask.empty() || mask.type() != CV_8UC1)
            return false;

        cv::Mat mask_f(mask.rows, mask.cols, CV_32FC1);
        const unsigned char* src = mask.data;
        float* dst = (float*)mask_f.data;
        const int total = mask.rows * mask.cols;
        for (int i = 0; i < total; i++)
            dst[i] = (float)src[i];

        cv::Mat scaled;
        if (!scale_mask(mask_f, target_h, target_w, scaled, padding))
            return false;

        cv::Mat bin(target_h, target_w, CV_8UC1);
        const float* sp = (const float*)scaled.data;
        unsigned char* bp = bin.data;
        const int out_total = target_h * target_w;
        for (int i = 0; i < out_total; i++)
            bp[i] = (sp[i] > 0.5f) ? 1 : 0;

        out_masks.push_back(std::move(bin));
    }

    return true;
}

inline bool process_mask_native(const ncnn::Mat& protos,
                                const ncnn::Mat& masks_in,
                                const std::vector<BoxXYXY>& bboxes_xyxy,
                                int shape_h,
                                int shape_w,
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

    out_masks.reserve(n);
    for (int i = 0; i < n; i++)
    {
        cv::Mat m = mask_logits.row(i).reshape(1, mh);  // (mh, mw) logits

        cv::Mat scaled;
        if (!scale_mask(m, shape_h, shape_w, scaled, true))
            return false;

        crop_mask_inplace((float*)scaled.data, shape_h, shape_w, bboxes_xyxy[i]);

        cv::Mat bin(shape_h, shape_w, CV_8UC1);
        const float* mp = (const float*)scaled.data;
        unsigned char* bp = bin.data;
        const int total = shape_h * shape_w;
        for (int j = 0; j < total; j++)
            bp[j] = (mp[j] > 0.f) ? 1 : 0;
        out_masks.push_back(std::move(bin));
    }

    return true;
}

}  // namespace yolo26
