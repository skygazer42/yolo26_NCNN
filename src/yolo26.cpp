#include "yolo26.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "cpu.h"

Yolo26::Yolo26(const Yolo26Config& config)
    : config_(config), net_(std::make_shared<ncnn::Net>())
{
}

Yolo26::~Yolo26() = default;

bool Yolo26::load(const std::string& param_path, const std::string& bin_path)
{
    if (!net_)
        net_ = std::make_shared<ncnn::Net>();

#if NCNN_VULKAN
    net_->opt.use_vulkan_compute = config_.use_gpu;
#endif
    net_->opt.num_threads = ncnn::get_big_cpu_count();

    if (net_->load_param(param_path.c_str()) != 0)
        return false;
    if (net_->load_model(bin_path.c_str()) != 0)
        return false;

    return true;
}

bool Yolo26::detect(const cv::Mat& bgr, std::vector<Yolo26Object>& objects) const
{
    if (!net_ || bgr.empty())
        return false;

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    float scale = std::min(config_.input_width / (float)img_w, config_.input_height / (float)img_h);
    int resized_w = std::max(1, (int)std::round(img_w * scale));
    int resized_h = std::max(1, (int)std::round(img_h * scale));

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, resized_w, resized_h);

    int pad_w = config_.input_width - resized_w;
    int pad_h = config_.input_height - resized_h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad,
                           pad_h / 2, pad_h - pad_h / 2,
                           pad_w / 2, pad_w - pad_w / 2,
                           ncnn::BORDER_CONSTANT, 0.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net_->create_extractor();
    int input_ret = ex.input(config_.input_name.c_str(), in_pad);
    if (input_ret != 0)
    {
        if (config_.input_name != "in0")
            input_ret = ex.input("in0", in_pad);
        if (config_.input_name != "images")
            input_ret = ex.input("images", in_pad);
        if (input_ret != 0 && config_.input_name != "data")
            input_ret = ex.input("data", in_pad);
        if (input_ret != 0)
            return false;
    }

    ncnn::Mat out;
    int out_ret = ex.extract(config_.output_name.c_str(), out);
    if (out_ret != 0 && config_.output_name != "out0")
        out_ret = ex.extract("out0", out);
    if (out_ret != 0 && config_.output_name != "output0")
        out_ret = ex.extract("output0", out);
    if (out_ret != 0 && config_.output_name != "output")
        out_ret = ex.extract("output", out);
    if (out_ret != 0)
        return false;

    ncnn::Mat out_2d = out;
    if (out.dims == 3)
        out_2d = out.channel(0);

    const int det_dim = 4 + config_.num_classes;

    // Ultralytics NCNN export (end2end disabled) typically outputs [4+nc, num_anchors] i.e. (84, 8400).
    // Apply Ultralytics-style TopK postprocess (no NMS) to match YOLO26 end-to-end behavior.
    if (out_2d.h == det_dim && out_2d.w > 0)
    {
        const int num_anchors = out_2d.w;
        const int k = std::max(1, std::min(config_.max_det, num_anchors));
        const float* box_cx = out_2d.row(0);
        const float* box_cy = out_2d.row(1);
        const float* box_w = out_2d.row(2);
        const float* box_h = out_2d.row(3);

        std::vector<const float*> score_rows(config_.num_classes);
        for (int c = 0; c < config_.num_classes; c++)
        {
            score_rows[c] = out_2d.row(4 + c);
        }

        struct AnchorScore {
            float score;
            int index;
        };
        std::vector<AnchorScore> anchor_scores;
        anchor_scores.reserve(num_anchors);
        for (int i = 0; i < num_anchors; i++)
        {
            float best = score_rows[0][i];
            for (int c = 1; c < config_.num_classes; c++)
                best = std::max(best, score_rows[c][i]);
            anchor_scores.push_back({best, i});
        }

        if (k < (int)anchor_scores.size())
        {
            std::nth_element(anchor_scores.begin(), anchor_scores.begin() + k, anchor_scores.end(),
                             [](const AnchorScore& a, const AnchorScore& b) { return a.score > b.score; });
            anchor_scores.resize(k);
        }
        std::sort(anchor_scores.begin(), anchor_scores.end(),
                  [](const AnchorScore& a, const AnchorScore& b) { return a.score > b.score; });

        struct Candidate {
            float score;
            int anchor;
            int cls;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(anchor_scores.size() * (size_t)config_.num_classes);
        for (const auto& a : anchor_scores)
        {
            for (int c = 0; c < config_.num_classes; c++)
            {
                candidates.push_back({score_rows[c][a.index], a.index, c});
            }
        }

        if (k < (int)candidates.size())
        {
            std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(),
                             [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
            candidates.resize(k);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        std::vector<Yolo26Object> proposals;
        proposals.reserve(candidates.size());
        for (const auto& cand : candidates)
        {
            if (cand.score < config_.conf_threshold)
                continue;

            const int i = cand.anchor;
            const float cx = box_cx[i];
            const float cy = box_cy[i];
            const float w = box_w[i];
            const float h = box_h[i];

            Yolo26Object obj;
            obj.rect.x = cx - w * 0.5f;
            obj.rect.y = cy - h * 0.5f;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.prob = cand.score;
            obj.label = cand.cls;
            proposals.push_back(obj);
        }

        objects.swap(proposals);
    }
    // Some converters may output [num_anchors, 4+nc] i.e. (8400, 84).
    else if (out_2d.w == det_dim && out_2d.h > 0)
    {
        const int num_anchors = out_2d.h;
        const int k = std::max(1, std::min(config_.max_det, num_anchors));

        struct AnchorScore {
            float score;
            int index;
        };
        std::vector<AnchorScore> anchor_scores;
        anchor_scores.reserve(num_anchors);
        for (int i = 0; i < num_anchors; i++)
        {
            const float* p = out_2d.row(i);
            float best = p[4];
            for (int c = 1; c < config_.num_classes; c++)
                best = std::max(best, p[4 + c]);
            anchor_scores.push_back({best, i});
        }

        if (k < (int)anchor_scores.size())
        {
            std::nth_element(anchor_scores.begin(), anchor_scores.begin() + k, anchor_scores.end(),
                             [](const AnchorScore& a, const AnchorScore& b) { return a.score > b.score; });
            anchor_scores.resize(k);
        }
        std::sort(anchor_scores.begin(), anchor_scores.end(),
                  [](const AnchorScore& a, const AnchorScore& b) { return a.score > b.score; });

        struct Candidate {
            float score;
            int anchor;
            int cls;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(anchor_scores.size() * (size_t)config_.num_classes);
        for (const auto& a : anchor_scores)
        {
            const float* p = out_2d.row(a.index);
            for (int c = 0; c < config_.num_classes; c++)
            {
                candidates.push_back({p[4 + c], a.index, c});
            }
        }

        if (k < (int)candidates.size())
        {
            std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(),
                             [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
            candidates.resize(k);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        std::vector<Yolo26Object> proposals;
        proposals.reserve(candidates.size());
        for (const auto& cand : candidates)
        {
            if (cand.score < config_.conf_threshold)
                continue;

            const float* p = out_2d.row(cand.anchor);
            const float cx = p[0];
            const float cy = p[1];
            const float w = p[2];
            const float h = p[3];

            Yolo26Object obj;
            obj.rect.x = cx - w * 0.5f;
            obj.rect.y = cy - h * 0.5f;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.prob = cand.score;
            obj.label = cand.cls;
            proposals.push_back(obj);
        }

        objects.swap(proposals);
    }
    // End-to-end export outputs (already top-k): [num_dets, 6] i.e. (300, 6) with xyxy + score + cls.
    else if (out_2d.w == 6 && out_2d.h > 0)
    {
        std::vector<Yolo26Object> proposals;
        const int num_dets = out_2d.h;
        proposals.reserve(num_dets);
        for (int i = 0; i < num_dets; i++)
        {
            const float* p = out_2d.row(i);
            float score = p[4];
            if (score < config_.conf_threshold)
                continue;

            Yolo26Object obj;
            obj.rect.x = p[0];
            obj.rect.y = p[1];
            obj.rect.width = p[2] - p[0];
            obj.rect.height = p[3] - p[1];
            obj.prob = score;
            obj.label = (int)p[5];
            proposals.push_back(obj);
        }

        objects.swap(proposals);
    }
    else
    {
        return false;
    }

    for (auto& obj : objects)
    {
        float x0 = (obj.rect.x - pad_w / 2.f) / scale;
        float y0 = (obj.rect.y - pad_h / 2.f) / scale;
        float x1 = (obj.rect.x + obj.rect.width - pad_w / 2.f) / scale;
        float y1 = (obj.rect.y + obj.rect.height - pad_h / 2.f) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
    }

    return true;
}
