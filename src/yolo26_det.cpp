#include "yolo26.h"
#include "yolo26_draw.h"

#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <cstdio>

static void print_usage(const char* prog)
{
    std::fprintf(stderr, "Usage: %s <param> <bin> <image> [output]\n", prog);
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
    const std::string output_path = (argc > 4) ? argv[4] : "yolo26_result.jpg";

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        std::fprintf(stderr, "Failed to read image: %s\n", image_path.c_str());
        return 1;
    }

    Yolo26 detector;
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
