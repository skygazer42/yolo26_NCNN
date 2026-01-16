#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "yolo26_mask.h"

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
                 "Usage: %s <n> <mask_dim> <mh> <mw> <in_h> <in_w> <orig_h> <orig_w> <seed> <out_dir>\n",
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

static bool write_u8(const std::string& path, const unsigned char* data, size_t count)
{
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    const size_t n = std::fwrite(data, sizeof(unsigned char), count, fp);
    std::fclose(fp);
    return n == count;
}

int main(int argc, char** argv)
{
    if (argc != 11)
    {
        print_usage(argv[0]);
        return 1;
    }

    const int n = std::atoi(argv[1]);
    const int mask_dim = std::atoi(argv[2]);
    const int mh = std::atoi(argv[3]);
    const int mw = std::atoi(argv[4]);
    const int in_h = std::atoi(argv[5]);
    const int in_w = std::atoi(argv[6]);
    const int orig_h = std::atoi(argv[7]);
    const int orig_w = std::atoi(argv[8]);
    const uint32_t seed = (uint32_t)std::strtoul(argv[9], 0, 10);
    const std::string out_dir = argv[10];

    if (n <= 0 || mask_dim <= 0 || mh <= 0 || mw <= 0 || in_h <= 0 || in_w <= 0 || orig_h <= 0 || orig_w <= 0)
        return 2;

    std::mt19937 rng(seed);
    std::normal_distribution<float> ndist(0.f, 1.f);
    std::uniform_real_distribution<float> udist01(0.f, 1.f);

    ncnn::Mat protos(mw, mh, mask_dim);
    for (int c = 0; c < mask_dim; c++)
    {
        float* p = protos.channel(c);
        for (int i = 0; i < mw * mh; i++)
            p[i] = ndist(rng);
    }

    ncnn::Mat masks_in(mask_dim, n);
    for (int i = 0; i < n; i++)
    {
        float* p = masks_in.row(i);
        for (int c = 0; c < mask_dim; c++)
            p[c] = ndist(rng);
    }

    std::vector<yolo26::BoxXYXY> boxes_in;
    std::vector<yolo26::BoxXYXY> boxes_orig;
    boxes_in.reserve(n);
    boxes_orig.reserve(n);
    std::vector<float> boxes_in_f32((size_t)n * 4);
    std::vector<float> boxes_orig_f32((size_t)n * 4);

    for (int i = 0; i < n; i++)
    {
        const float x1 = udist01(rng) * (in_w - 1);
        const float y1 = udist01(rng) * (in_h - 1);
        const float x2 = x1 + udist01(rng) * (in_w - x1);
        const float y2 = y1 + udist01(rng) * (in_h - y1);
        yolo26::BoxXYXY b_in;
        b_in.x1 = std::min(x1, x2);
        b_in.y1 = std::min(y1, y2);
        b_in.x2 = std::max(x1, x2);
        b_in.y2 = std::max(y1, y2);
        boxes_in.push_back(b_in);
        boxes_in_f32[(size_t)i * 4 + 0] = b_in.x1;
        boxes_in_f32[(size_t)i * 4 + 1] = b_in.y1;
        boxes_in_f32[(size_t)i * 4 + 2] = b_in.x2;
        boxes_in_f32[(size_t)i * 4 + 3] = b_in.y2;

        const float ox1 = udist01(rng) * (orig_w - 1);
        const float oy1 = udist01(rng) * (orig_h - 1);
        const float ox2 = ox1 + udist01(rng) * (orig_w - ox1);
        const float oy2 = oy1 + udist01(rng) * (orig_h - oy1);
        yolo26::BoxXYXY b0;
        b0.x1 = std::min(ox1, ox2);
        b0.y1 = std::min(oy1, oy2);
        b0.x2 = std::max(ox1, ox2);
        b0.y2 = std::max(oy1, oy2);
        boxes_orig.push_back(b0);
        boxes_orig_f32[(size_t)i * 4 + 0] = b0.x1;
        boxes_orig_f32[(size_t)i * 4 + 1] = b0.y1;
        boxes_orig_f32[(size_t)i * 4 + 2] = b0.x2;
        boxes_orig_f32[(size_t)i * 4 + 3] = b0.y2;
    }

    std::vector<cv::Mat> masks_proc;
    if (!yolo26::process_mask(protos, masks_in, boxes_in, in_h, in_w, true, masks_proc))
        return 3;

    std::vector<cv::Mat> masks_scaled;
    if (!yolo26::scale_masks(masks_proc, orig_h, orig_w, masks_scaled, true))
        return 4;

    std::vector<cv::Mat> masks_native;
    if (!yolo26::process_mask_native(protos, masks_in, boxes_orig, orig_h, orig_w, masks_native))
        return 5;

    const std::string protos_path = out_dir + "/protos.bin";
    const std::string masks_in_path = out_dir + "/masks_in.bin";
    const std::string boxes_in_path = out_dir + "/boxes_in.bin";
    const std::string boxes_orig_path = out_dir + "/boxes_orig.bin";
    const std::string masks_proc_path = out_dir + "/masks_proc.bin";
    const std::string masks_scaled_path = out_dir + "/masks_scaled.bin";
    const std::string masks_native_path = out_dir + "/masks_native.bin";

    if (!write_f32(protos_path, (const float*)protos.data, (size_t)mask_dim * (size_t)mh * (size_t)mw))
        return 6;
    if (!write_f32(masks_in_path, (const float*)masks_in.data, (size_t)n * (size_t)mask_dim))
        return 7;
    if (!write_f32(boxes_in_path, boxes_in_f32.data(), boxes_in_f32.size()))
        return 8;
    if (!write_f32(boxes_orig_path, boxes_orig_f32.data(), boxes_orig_f32.size()))
        return 9;

    {
        std::vector<unsigned char> buf((size_t)n * (size_t)in_h * (size_t)in_w);
        size_t offset = 0;
        for (const auto& m : masks_proc)
        {
            if (m.empty() || m.type() != CV_8UC1 || m.rows != in_h || m.cols != in_w || !m.isContinuous())
                return 10;
            std::copy(m.data, m.data + (size_t)in_h * (size_t)in_w, buf.data() + offset);
            offset += (size_t)in_h * (size_t)in_w;
        }
        if (!write_u8(masks_proc_path, buf.data(), buf.size()))
            return 11;
    }

    {
        std::vector<unsigned char> buf((size_t)n * (size_t)orig_h * (size_t)orig_w);
        size_t offset = 0;
        for (const auto& m : masks_scaled)
        {
            if (m.empty() || m.type() != CV_8UC1 || m.rows != orig_h || m.cols != orig_w || !m.isContinuous())
                return 12;
            std::copy(m.data, m.data + (size_t)orig_h * (size_t)orig_w, buf.data() + offset);
            offset += (size_t)orig_h * (size_t)orig_w;
        }
        if (!write_u8(masks_scaled_path, buf.data(), buf.size()))
            return 13;
    }

    {
        std::vector<unsigned char> buf((size_t)n * (size_t)orig_h * (size_t)orig_w);
        size_t offset = 0;
        for (const auto& m : masks_native)
        {
            if (m.empty() || m.type() != CV_8UC1 || m.rows != orig_h || m.cols != orig_w || !m.isContinuous())
                return 14;
            std::copy(m.data, m.data + (size_t)orig_h * (size_t)orig_w, buf.data() + offset);
            offset += (size_t)orig_h * (size_t)orig_w;
        }
        if (!write_u8(masks_native_path, buf.data(), buf.size()))
            return 15;
    }

    return 0;
}

