#pragma once

#include "yolo26_types.h"

#include <cstdlib>
#include <string>

namespace yolo26_cli {

inline bool starts_with(const std::string& s, const char* prefix)
{
    return s.rfind(prefix, 0) == 0;
}

inline bool parse_float(const char* s, float& out)
{
    if (!s || !*s)
        return false;
    char* end = 0;
    out = std::strtof(s, &end);
    return end && *end == '\0';
}

inline bool parse_int(const char* s, int& out)
{
    if (!s || !*s)
        return false;
    char* end = 0;
    long v = std::strtol(s, &end, 10);
    if (!(end && *end == '\0'))
        return false;
    out = (int)v;
    return true;
}

inline bool parse_common_arg(const std::string& arg,
                             int argc,
                             char** argv,
                             int& argi,
                             float& conf_threshold,
                             float& iou_threshold,
                             int& max_det,
                             Yolo26PostprocessType& postprocess,
                             Yolo26BoxFormat& box_format,
                             bool& topk_dedup,
                             bool& agnostic_nms,
                             bool& use_gpu)
{
    if (arg == "--agnostic")
    {
        agnostic_nms = true;
        return true;
    }
    if (arg == "--dedup")
    {
        topk_dedup = true;
        return true;
    }
    if (arg == "--gpu")
    {
        use_gpu = true;
        return true;
    }

    if (arg == "--conf" || starts_with(arg, "--conf="))
    {
        const char* v = 0;
        if (arg == "--conf")
        {
            if (argi >= argc)
                return false;
            v = argv[argi++];
        }
        else
        {
            v = arg.c_str() + std::string("--conf=").size();
        }
        float f = 0.f;
        if (!parse_float(v, f))
            return false;
        conf_threshold = f;
        return true;
    }

    if (arg == "--iou" || starts_with(arg, "--iou="))
    {
        const char* v = 0;
        if (arg == "--iou")
        {
            if (argi >= argc)
                return false;
            v = argv[argi++];
        }
        else
        {
            v = arg.c_str() + std::string("--iou=").size();
        }
        float f = 0.f;
        if (!parse_float(v, f))
            return false;
        iou_threshold = f;
        return true;
    }

    if (arg == "--max-det" || starts_with(arg, "--max-det="))
    {
        const char* v = 0;
        if (arg == "--max-det")
        {
            if (argi >= argc)
                return false;
            v = argv[argi++];
        }
        else
        {
            v = arg.c_str() + std::string("--max-det=").size();
        }
        int n = 0;
        if (!parse_int(v, n))
            return false;
        max_det = n;
        return true;
    }

    if (arg == "--post" || starts_with(arg, "--post="))
    {
        const char* v = 0;
        if (arg == "--post")
        {
            if (argi >= argc)
                return false;
            v = argv[argi++];
        }
        else
        {
            v = arg.c_str() + std::string("--post=").size();
        }
        const std::string mode = v ? v : "";
        if (mode == "auto")
            postprocess = Yolo26PostprocessType::Auto;
        else if (mode == "nms")
            postprocess = Yolo26PostprocessType::NMS;
        else if (mode == "topk")
            postprocess = Yolo26PostprocessType::TopK;
        else
            return false;
        return true;
    }

    if (arg == "--box" || starts_with(arg, "--box="))
    {
        const char* v = 0;
        if (arg == "--box")
        {
            if (argi >= argc)
                return false;
            v = argv[argi++];
        }
        else
        {
            v = arg.c_str() + std::string("--box=").size();
        }
        const std::string mode = v ? v : "";
        if (mode == "cxcywh")
            box_format = Yolo26BoxFormat::CXCYWH;
        else if (mode == "xyxy")
            box_format = Yolo26BoxFormat::XYXY;
        else
            return false;
        return true;
    }

    return false;
}

}  // namespace yolo26_cli

