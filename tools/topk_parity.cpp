#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include "yolo26_topk.h"

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
                 "Usage: %s <anchors> <classes> <max_det> <seed> <scores.bin> <topk.txt>\n",
                 prog);
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        print_usage(argv[0]);
        return 1;
    }

    const int anchors = std::atoi(argv[1]);
    const int classes = std::atoi(argv[2]);
    const int max_det = std::atoi(argv[3]);
    const uint32_t seed = (uint32_t)std::strtoul(argv[4], 0, 10);
    const std::string scores_path = argv[5];
    const std::string topk_path = argv[6];

    if (anchors <= 0 || classes <= 0 || max_det <= 0)
        return 2;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    std::vector<float> scores((size_t)anchors * (size_t)classes);
    for (size_t i = 0; i < scores.size(); i++)
        scores[i] = dist(rng);

    {
        std::FILE* fp = std::fopen(scores_path.c_str(), "wb");
        if (!fp)
            return 3;
        const size_t n = std::fwrite(scores.data(), sizeof(float), scores.size(), fp);
        std::fclose(fp);
        if (n != scores.size())
            return 4;
    }

    const auto topk = yolo26::get_topk_index(
        anchors,
        classes,
        max_det,
        [&](int anchor, int cls) { return scores[(size_t)anchor * (size_t)classes + (size_t)cls]; });

    std::ofstream ofs(topk_path.c_str(), std::ios::out);
    if (!ofs.is_open())
        return 5;
    ofs.setf(std::ios::fixed);
    ofs << std::setprecision(9);
    for (const auto& r : topk)
        ofs << r.score << " " << r.cls << " " << r.anchor << "\n";

    return 0;
}

