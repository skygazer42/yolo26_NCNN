#pragma once

enum class Yolo26BoxFormat
{
    CXCYWH = 0,
    XYXY = 1,
};

enum class Yolo26PostprocessType
{
    Auto = 0,
    NMS = 1,
    TopK = 2,
};
