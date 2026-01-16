#include "yolo26.h"
#include "yolo26_draw.h"
#include "yolo26_cli.h"

#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <cstdio>
#include <string>

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
                 "  --dedup                  Apply IoU de-dup after TopK\n"
                 "  --agnostic               Class-agnostic NMS\n"
                 "  --gpu                    Enable Vulkan (if available)\n",
                 prog);
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
    std::string output_path = "yolo26_result.jpg";
    if (argi < argc && !yolo26_cli::starts_with(argv[argi], "--"))
        output_path = argv[argi++];

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        std::fprintf(stderr, "Failed to read image: %s\n", image_path.c_str());
        return 1;
    }

    Yolo26Config config;
    while (argi < argc)
    {
        const std::string arg = argv[argi++];
        if (!yolo26_cli::parse_common_arg(arg,
                                          argc,
                                          argv,
                                          argi,
                                          config.conf_threshold,
                                          config.iou_threshold,
                                          config.max_det,
                                          config.postprocess,
                                          config.box_format,
                                          config.topk_dedup,
                                          config.agnostic_nms,
                                          config.use_gpu))
            return (print_usage(argv[0]), 1);
    }
    Yolo26 detector(config);
    if (!detector.load(param_path, bin_path))
    {
        std::fprintf(stderr, "Failed to load model: %s %s\n", param_path.c_str(), bin_path.c_str());
        return 1;
    }

    std::vector<Yolo26Object> objects;
    if (!detector.detect(bgr, objects))
    {
        std::fprintf(stderr, "Detection failed\n");
        return 1;
    }

    yolo26_draw_objects(bgr, objects);

    if (!cv::imwrite(output_path, bgr))
    {
        std::fprintf(stderr, "Failed to write output: %s\n", output_path.c_str());
        return 1;
    }

    std::fprintf(stdout, "Saved: %s (objects=%zu)\n", output_path.c_str(), objects.size());
    return 0;
}
