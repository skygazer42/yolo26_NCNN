#include "yolo26_draw.h"

#include <opencv2/imgproc/imgproc.hpp>

const std::vector<std::string>& yolo26_coco_names()
{
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    return names;
}

const std::vector<cv::Scalar>& yolo26_coco_colors()
{
    static const std::vector<cv::Scalar> colors = {
        {56, 0, 255}, {226, 255, 0}, {0, 94, 255}, {0, 37, 255}, {0, 255, 94},
        {255, 226, 0}, {0, 18, 255}, {255, 151, 0}, {170, 0, 255}, {0, 255, 56},
        {255, 0, 75}, {0, 75, 255}, {0, 255, 169}, {255, 0, 207}, {75, 255, 0},
        {207, 0, 255}, {37, 0, 255}, {0, 207, 255}, {94, 0, 255}, {0, 255, 113},
        {255, 18, 0}, {255, 0, 56}, {18, 0, 255}, {0, 255, 226}, {170, 255, 0},
        {255, 0, 245}, {151, 255, 0}, {132, 255, 0}, {75, 0, 255}, {151, 0, 255},
        {0, 151, 255}, {132, 0, 255}, {0, 255, 245}, {255, 132, 0}, {226, 0, 255},
        {255, 37, 0}, {207, 255, 0}, {0, 255, 207}, {94, 255, 0}, {0, 226, 255},
        {56, 255, 0}, {255, 94, 0}, {255, 113, 0}, {0, 132, 255}, {255, 0, 132},
        {255, 170, 0}, {255, 0, 188}, {113, 255, 0}, {245, 0, 255}, {113, 0, 255},
        {255, 188, 0}, {0, 113, 255}, {255, 0, 0}, {0, 56, 255}, {255, 0, 113},
        {0, 255, 188}, {255, 0, 94}, {255, 0, 18}, {18, 255, 0}, {0, 255, 132},
        {0, 188, 255}, {0, 245, 255}, {0, 169, 255}, {37, 255, 0}, {255, 0, 151},
        {188, 0, 255}, {0, 255, 37}, {0, 255, 0}, {255, 0, 170}, {255, 0, 37},
        {255, 75, 0}, {0, 0, 255}, {255, 207, 0}, {255, 0, 226}, {255, 245, 0},
        {188, 255, 0}, {0, 255, 18}, {0, 255, 75}, {0, 255, 151}, {255, 56, 0},
        {245, 255, 0}, {255, 0, 245}
    };
    return colors;
}

void yolo26_draw_objects(cv::Mat& bgr, const std::vector<Yolo26Object>& objects, float font_scale)
{
    const auto& names = yolo26_coco_names();
    const auto& colors = yolo26_coco_colors();

    for (const auto& obj : objects)
    {
        int label = obj.label;
        if (label < 0)
            label = 0;
        cv::Scalar color = colors[label % colors.size()];
        cv::Rect_<float> rect(cv::Point2f(obj.x1, obj.y1), cv::Point2f(obj.x2, obj.y2));
        cv::rectangle(bgr, rect, color, 2);

        std::string label_text;
        if (obj.label >= 0 && obj.label < (int)names.size())
            label_text = names[obj.label];
        else
            label_text = cv::format("cls%d", obj.label);
        label_text += cv::format(" %.2f", obj.prob);
        int base_line = 0;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &base_line);

        float x = obj.x1;
        float y = obj.y1 - label_size.height - base_line;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect_<float>(cv::Point2f(x, y),
                                            cv::Size_<float>(label_size.width, label_size.height + base_line)),
                      color, -1);

        cv::putText(bgr, label_text, cv::Point2f(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), 1);
    }
}
