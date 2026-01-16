#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include "yolo26_nms.h"

struct Det {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    int label = -1;
    float prob = 0.f;
};

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
                 "Usage: %s <anchors> <classes> <max_det> <conf> <iou> <agnostic:0|1> <seed> <out_dir>\n",
                 prog);
}

static bool write_f32(const std::string& path, const float* data, size_t count)
{
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    const size_t n = std::fwrite(data, sizeof(float), count, fp);
    std::fclose(fp);
    return n == count;
}

int main(int argc, char** argv)
{
    if (argc != 9)
    {
        print_usage(argv[0]);
        return 1;
    }

    const int anchors = std::atoi(argv[1]);
    const int classes = std::atoi(argv[2]);
    const int max_det = std::atoi(argv[3]);
    const float conf_thres = std::strtof(argv[4], 0);
    const float iou_thres = std::strtof(argv[5], 0);
    const bool agnostic = std::atoi(argv[6]) != 0;
    const uint32_t seed = (uint32_t)std::strtoul(argv[7], 0, 10);
    const std::string out_dir = argv[8];

    if (anchors <= 0 || classes <= 0 || max_det <= 0)
        return 2;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist01(0.f, 1.f);

    std::vector<float> boxes_cxcywh((size_t)anchors * 4);
    for (int i = 0; i < anchors; i++)
    {
        const float x1 = dist01(rng) * 640.f;
        const float y1 = dist01(rng) * 640.f;
        const float x2 = x1 + dist01(rng) * (640.f - x1);
        const float y2 = y1 + dist01(rng) * (640.f - y1);
        const float cx = (x1 + x2) * 0.5f;
        const float cy = (y1 + y2) * 0.5f;
        const float w = (x2 - x1);
        const float h = (y2 - y1);
        boxes_cxcywh[(size_t)i * 4 + 0] = cx;
        boxes_cxcywh[(size_t)i * 4 + 1] = cy;
        boxes_cxcywh[(size_t)i * 4 + 2] = w;
        boxes_cxcywh[(size_t)i * 4 + 3] = h;
    }

    std::vector<float> scores((size_t)anchors * (size_t)classes);
    for (size_t i = 0; i < scores.size(); i++)
        scores[i] = dist01(rng);

    std::vector<Det> proposals;
    proposals.reserve((size_t)anchors);
    for (int a = 0; a < anchors; a++)
    {
        const float* sp = scores.data() + (size_t)a * (size_t)classes;
        float best = sp[0];
        int best_cls = 0;
        for (int c = 1; c < classes; c++)
        {
            const float s = sp[c];
            if (s > best)
            {
                best = s;
                best_cls = c;
            }
        }
        if (best < conf_thres)
            continue;

        const float cx = boxes_cxcywh[(size_t)a * 4 + 0];
        const float cy = boxes_cxcywh[(size_t)a * 4 + 1];
        const float w = boxes_cxcywh[(size_t)a * 4 + 2];
        const float h = boxes_cxcywh[(size_t)a * 4 + 3];

        Det d;
        d.x1 = cx - w * 0.5f;
        d.y1 = cy - h * 0.5f;
        d.x2 = cx + w * 0.5f;
        d.y2 = cy + h * 0.5f;
        d.prob = best;
        d.label = best_cls;
        proposals.push_back(d);
    }

    auto kept = yolo26::nms(proposals, iou_thres, agnostic);
    if ((int)kept.size() > max_det)
        kept.resize((size_t)max_det);

    const std::string boxes_path = out_dir + "/boxes_cxcywh.bin";
    const std::string scores_path = out_dir + "/scores.bin";
    const std::string dets_path = out_dir + "/dets.txt";

    if (!write_f32(boxes_path, boxes_cxcywh.data(), boxes_cxcywh.size()))
        return 3;
    if (!write_f32(scores_path, scores.data(), scores.size()))
        return 4;

    std::ofstream ofs(dets_path.c_str(), std::ios::out);
    if (!ofs.is_open())
        return 5;
    ofs.setf(std::ios::fixed);
    ofs << std::setprecision(9);
    for (const auto& d : kept)
        ofs << d.x1 << " " << d.y1 << " " << d.x2 << " " << d.y2 << " " << d.prob << " " << d.label << "\n";

    return 0;
}

