#include "yolo26_draw.h"
#include "yolo26_seg.h"

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

static bool starts_with(const std::string& s, const char* prefix)
{
    return s.rfind(prefix, 0) == 0;
}

static bool parse_float(const char* s, float& out)
{
    if (!s || !*s)
        return false;
    char* end = 0;
    out = std::strtof(s, &end);
    return end && *end == '\0';
}

static bool parse_int(const char* s, int& out)
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

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
                 "Usage: %s <param> <bin> <image> [output] [options]\n"
                 "\n"
                 "Options:\n"
                 "  --conf <float>           Confidence threshold\n"
                 "  --iou <float>            IoU threshold (NMS)\n"
                 "  --max-det <int>          Max detections\n"
                 "  --post <auto|nms|topk>    Postprocess mode\n"
                 "  --box <cxcywh|xyxy>       Box format for raw outputs\n"
                 "  --agnostic               Class-agnostic NMS\n"
                 "  --retina                 Use retina masks path\n"
                 "  --gpu                    Enable Vulkan (if available)\n",
                 prog);
}

static void draw_segmentation(cv::Mat& bgr, const std::vector<Yolo26SegObject>& objects, float mask_thresh)
{
    const auto& colors = yolo26_coco_colors();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const auto& obj = objects[i];
        cv::Scalar color = colors[obj.label % colors.size()];

        if (obj.mask.empty() || obj.mask.type() != CV_8UC1)
            continue;
        (void)mask_thresh;
        const cv::Mat& mask_bin = obj.mask;

        cv::Mat overlay = bgr.clone();
        overlay.setTo(color, mask_bin);

        cv::Mat blended;
        cv::addWeighted(overlay, 0.5, bgr, 0.5, 0, blended);
        blended.copyTo(bgr, mask_bin);
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        print_usage(argv[0]);
        return 1;
    }

    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string image_path = argv[3];
    int argi = 4;
    std::string output_path = "yolo26_seg_result.jpg";
    if (argi < argc && !starts_with(argv[argi], "--"))
        output_path = argv[argi++];

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        std::fprintf(stderr, "Failed to read image: %s\n", image_path.c_str());
        return 1;
    }

    Yolo26SegConfig config;
    while (argi < argc)
    {
        const std::string arg = argv[argi++];
        if (arg == "--agnostic")
        {
            config.agnostic_nms = true;
        }
        else if (arg == "--retina")
        {
            config.retina_masks = true;
        }
        else if (arg == "--gpu")
        {
            config.use_gpu = true;
        }
        else if (arg == "--conf" || starts_with(arg, "--conf="))
        {
            const char* v = 0;
            if (arg == "--conf")
            {
                if (argi >= argc)
                    return (print_usage(argv[0]), 1);
                v = argv[argi++];
            }
            else
            {
                v = arg.c_str() + std::string("--conf=").size();
            }
            float f = 0.f;
            if (!parse_float(v, f))
                return (print_usage(argv[0]), 1);
            config.conf_threshold = f;
        }
        else if (arg == "--iou" || starts_with(arg, "--iou="))
        {
            const char* v = 0;
            if (arg == "--iou")
            {
                if (argi >= argc)
                    return (print_usage(argv[0]), 1);
                v = argv[argi++];
            }
            else
            {
                v = arg.c_str() + std::string("--iou=").size();
            }
            float f = 0.f;
            if (!parse_float(v, f))
                return (print_usage(argv[0]), 1);
            config.iou_threshold = f;
        }
        else if (arg == "--max-det" || starts_with(arg, "--max-det="))
        {
            const char* v = 0;
            if (arg == "--max-det")
            {
                if (argi >= argc)
                    return (print_usage(argv[0]), 1);
                v = argv[argi++];
            }
            else
            {
                v = arg.c_str() + std::string("--max-det=").size();
            }
            int n = 0;
            if (!parse_int(v, n))
                return (print_usage(argv[0]), 1);
            config.max_det = n;
        }
        else if (arg == "--post" || starts_with(arg, "--post="))
        {
            const char* v = 0;
            if (arg == "--post")
            {
                if (argi >= argc)
                    return (print_usage(argv[0]), 1);
                v = argv[argi++];
            }
            else
            {
                v = arg.c_str() + std::string("--post=").size();
            }
            const std::string mode = v ? v : "";
            if (mode == "auto")
                config.postprocess = Yolo26PostprocessType::Auto;
            else if (mode == "nms")
                config.postprocess = Yolo26PostprocessType::NMS;
            else if (mode == "topk")
                config.postprocess = Yolo26PostprocessType::TopK;
            else
                return (print_usage(argv[0]), 1);
        }
        else if (arg == "--box" || starts_with(arg, "--box="))
        {
            const char* v = 0;
            if (arg == "--box")
            {
                if (argi >= argc)
                    return (print_usage(argv[0]), 1);
                v = argv[argi++];
            }
            else
            {
                v = arg.c_str() + std::string("--box=").size();
            }
            const std::string mode = v ? v : "";
            if (mode == "cxcywh")
                config.box_format = Yolo26BoxFormat::CXCYWH;
            else if (mode == "xyxy")
                config.box_format = Yolo26BoxFormat::XYXY;
            else
                return (print_usage(argv[0]), 1);
        }
        else
        {
            return (print_usage(argv[0]), 1);
        }
    }

    Yolo26Seg detector(config);
    if (!detector.load(param_path, bin_path))
    {
        std::fprintf(stderr, "Failed to load model: %s %s\n", param_path.c_str(), bin_path.c_str());
        return 1;
    }

    std::vector<Yolo26SegObject> objects;
    if (!detector.detect(bgr, objects))
    {
        std::fprintf(stderr, "Segmentation failed\n");
        return 1;
    }

    draw_segmentation(bgr, objects, detector.config().mask_threshold);
    std::vector<Yolo26Object> det_objects;
    det_objects.reserve(objects.size());
    for (const auto& obj : objects)
    {
        Yolo26Object det;
        det.x1 = obj.x1;
        det.y1 = obj.y1;
        det.x2 = obj.x2;
        det.y2 = obj.y2;
        det.label = obj.label;
        det.prob = obj.prob;
        det_objects.push_back(det);
    }
    yolo26_draw_objects(bgr, det_objects);

    if (!cv::imwrite(output_path, bgr))
    {
        std::fprintf(stderr, "Failed to write output: %s\n", output_path.c_str());
        return 1;
    }

    std::fprintf(stdout, "Saved: %s (objects=%zu)\n", output_path.c_str(), objects.size());
    return 0;
}
