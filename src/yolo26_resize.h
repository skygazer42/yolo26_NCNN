#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace yolo26 {

inline void resize_bilinear_align_false(const float* src, int src_h, int src_w, float* dst, int dst_h, int dst_w)
{
    if (!src || !dst || src_h <= 0 || src_w <= 0 || dst_h <= 0 || dst_w <= 0)
        return;

    const float scale_y = (float)src_h / (float)dst_h;
    const float scale_x = (float)src_w / (float)dst_w;

    for (int dy = 0; dy < dst_h; dy++)
    {
        const float fy = (dy + 0.5f) * scale_y - 0.5f;
        int y0 = (int)std::floor(fy);
        int y1 = y0 + 1;
        const float ly = fy - y0;
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));

        for (int dx = 0; dx < dst_w; dx++)
        {
            const float fx = (dx + 0.5f) * scale_x - 0.5f;
            int x0 = (int)std::floor(fx);
            int x1 = x0 + 1;
            const float lx = fx - x0;
            x0 = std::max(0, std::min(x0, src_w - 1));
            x1 = std::max(0, std::min(x1, src_w - 1));

            const float v00 = src[y0 * src_w + x0];
            const float v01 = src[y0 * src_w + x1];
            const float v10 = src[y1 * src_w + x0];
            const float v11 = src[y1 * src_w + x1];

            const float v0 = v00 + (v01 - v00) * lx;
            const float v1 = v10 + (v11 - v10) * lx;
            dst[dy * dst_w + dx] = v0 + (v1 - v0) * ly;
        }
    }
}

inline void resize_bilinear_align_false(const std::vector<float>& src, int src_h, int src_w,
                                        std::vector<float>& dst, int dst_h, int dst_w)
{
    dst.assign((size_t)dst_h * (size_t)dst_w, 0.f);
    resize_bilinear_align_false(src.data(), src_h, src_w, dst.data(), dst_h, dst_w);
}

}  // namespace yolo26

