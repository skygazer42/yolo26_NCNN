#pragma once

#include <algorithm>

#include "yolo26_preprocess.h"

namespace yolo26 {

inline float clampf(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
}

inline void scale_xyxy_inplace(float& x1,
                               float& y1,
                               float& x2,
                               float& y2,
                               int img0_w,
                               int img0_h,
                               const LetterBoxInfo& lb,
                               bool padding = true)
{
    if (lb.gain <= 0.f)
        return;
    if (padding)
    {
        x1 -= lb.pad_x;
        x2 -= lb.pad_x;
        y1 -= lb.pad_y;
        y2 -= lb.pad_y;
    }

    x1 /= lb.gain;
    x2 /= lb.gain;
    y1 /= lb.gain;
    y2 /= lb.gain;

    x1 = clampf(x1, 0.f, (float)img0_w);
    y1 = clampf(y1, 0.f, (float)img0_h);
    x2 = clampf(x2, 0.f, (float)img0_w);
    y2 = clampf(y2, 0.f, (float)img0_h);
}

}  // namespace yolo26

