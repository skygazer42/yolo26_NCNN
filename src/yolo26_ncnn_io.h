#pragma once

#include <initializer_list>
#include <string>

#include "mat.h"
#include "net.h"

namespace yolo26 {

inline bool ncnn_input_with_fallback(ncnn::Extractor& ex,
                                    const std::string& preferred,
                                    const ncnn::Mat& in,
                                    std::initializer_list<const char*> fallbacks)
{
    int ret = ex.input(preferred.c_str(), in);
    if (ret == 0)
        return true;
    for (const char* name : fallbacks)
    {
        if (!name || preferred == name)
            continue;
        ret = ex.input(name, in);
        if (ret == 0)
            return true;
    }
    return false;
}

inline bool ncnn_input_image(ncnn::Extractor& ex, const std::string& preferred, const ncnn::Mat& in)
{
    return ncnn_input_with_fallback(ex, preferred, in, {"in0", "images", "data"});
}

inline bool ncnn_extract_with_fallback(ncnn::Extractor& ex,
                                      const std::string& preferred,
                                      ncnn::Mat& out,
                                      std::initializer_list<const char*> fallbacks)
{
    int ret = ex.extract(preferred.c_str(), out);
    if (ret == 0)
        return true;
    for (const char* name : fallbacks)
    {
        if (!name || preferred == name)
            continue;
        ret = ex.extract(name, out);
        if (ret == 0)
            return true;
    }
    return false;
}

inline bool ncnn_extract_out0(ncnn::Extractor& ex, const std::string& preferred, ncnn::Mat& out)
{
    return ncnn_extract_with_fallback(ex, preferred, out, {"out0", "output0", "output"});
}

inline bool ncnn_extract_out1(ncnn::Extractor& ex, const std::string& preferred, ncnn::Mat& out)
{
    return ncnn_extract_with_fallback(ex, preferred, out, {"out1", "seg", "output1"});
}

}  // namespace yolo26
