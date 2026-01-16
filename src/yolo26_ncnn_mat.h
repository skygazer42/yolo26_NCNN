#pragma once

#include "mat.h"

namespace yolo26 {

inline bool to_mat2d(const ncnn::Mat& in, ncnn::Mat& out)
{
    if (in.dims == 2)
    {
        out = in;
        return true;
    }

    if (in.dims == 3)
    {
        if (in.c == 1)
        {
            out = in.channel(0);
            return true;
        }
        // Common layout for CHW tensors exported as (w, 1, c): treat channels as rows.
        if (in.h == 1)
        {
            out = in.reshape(in.w, in.c);
            return true;
        }
        // Some converters may export as (1, h, c): treat channels as rows.
        if (in.w == 1)
        {
            out = in.reshape(in.h, in.c);
            return true;
        }
    }

    return false;
}

}  // namespace yolo26

