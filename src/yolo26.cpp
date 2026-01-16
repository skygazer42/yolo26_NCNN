#include "yolo26.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "cpu.h"

#include "yolo26_preprocess.h"
#include "yolo26_ops.h"
#include "yolo26_ncnn_mat.h"
#include "yolo26_topk.h"
#include "yolo26_nms.h"

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

    yolo26::LetterBoxInfo lb;
    ncnn::Mat in_pad;
    if (!yolo26::letterbox(bgr,
                           config_.input_width,
                           config_.input_height,
                           config_.padding_value,
                           config_.scaleup,
                           config_.center,
                           in_pad,
                           lb))
        return false;
    yolo26::normalize_01_inplace(in_pad);

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

    ncnn::Mat out_2d;
    if (!yolo26::to_mat2d(out, out_2d))
        return false;

    const int det_dim = 4 + config_.num_classes;

    // Ultralytics NCNN export (end2end disabled) typically outputs [4+nc, num_anchors] i.e. (84, 8400).
    // Apply Ultralytics-style TopK postprocess (no NMS) to match YOLO26 end-to-end behavior.
    if (out_2d.h == det_dim && out_2d.w > 0)
    {
        const int num_anchors = out_2d.w;
        const float* box_cx = out_2d.row(0);
        const float* box_cy = out_2d.row(1);
        const float* box_w = out_2d.row(2);
        const float* box_h = out_2d.row(3);

        std::vector<const float*> score_rows(config_.num_classes);
        for (int c = 0; c < config_.num_classes; c++)
        {
            score_rows[c] = out_2d.row(4 + c);
        }

        const auto topk = yolo26::get_topk_index(
            num_anchors, config_.num_classes, config_.max_det,
            [&](int anchor, int cls) { return score_rows[cls][anchor]; });

        std::vector<Yolo26Object> proposals;
        proposals.reserve(topk.size());
        for (const auto& cand : topk)
        {
            if (cand.score < config_.conf_threshold)
                continue;

            const int i = cand.anchor;
            const float cx = box_cx[i];
            const float cy = box_cy[i];
            const float w = box_w[i];
            const float h = box_h[i];

            Yolo26Object obj;
            obj.x1 = cx - w * 0.5f;
            obj.y1 = cy - h * 0.5f;
            obj.x2 = cx + w * 0.5f;
            obj.y2 = cy + h * 0.5f;
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

        const auto topk = yolo26::get_topk_index(
            num_anchors, config_.num_classes, config_.max_det,
            [&](int anchor, int cls) { return out_2d.row(anchor)[4 + cls]; });

        std::vector<Yolo26Object> proposals;
        proposals.reserve(topk.size());
        for (const auto& cand : topk)
        {
            if (cand.score < config_.conf_threshold)
                continue;

            const float* p = out_2d.row(cand.anchor);
            const float cx = p[0];
            const float cy = p[1];
            const float w = p[2];
            const float h = p[3];

            Yolo26Object obj;
            obj.x1 = cx - w * 0.5f;
            obj.y1 = cy - h * 0.5f;
            obj.x2 = cx + w * 0.5f;
            obj.y2 = cy + h * 0.5f;
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
        proposals.reserve(std::min(num_dets, config_.max_det));
        for (int i = 0; i < num_dets; i++)
        {
            const float* p = out_2d.row(i);
            float score = p[4];
            if (score < config_.conf_threshold)
                continue;

            Yolo26Object obj;
            obj.x1 = p[0];
            obj.y1 = p[1];
            obj.x2 = p[2];
            obj.y2 = p[3];
            obj.prob = score;
            obj.label = (int)p[5];
            proposals.push_back(obj);
            if ((int)proposals.size() >= config_.max_det)
                break;
        }

        objects.swap(proposals);
    }
    // Some converters may output [6, num_dets] i.e. (6, 300).
    else if (out_2d.h == 6 && out_2d.w > 0)
    {
        const int num_dets = out_2d.w;
        const float* x1_row = out_2d.row(0);
        const float* y1_row = out_2d.row(1);
        const float* x2_row = out_2d.row(2);
        const float* y2_row = out_2d.row(3);
        const float* score_row = out_2d.row(4);
        const float* cls_row = out_2d.row(5);

        std::vector<Yolo26Object> proposals;
        proposals.reserve(std::min(num_dets, config_.max_det));
        for (int i = 0; i < num_dets; i++)
        {
            const float score = score_row[i];
            if (score < config_.conf_threshold)
                continue;

            Yolo26Object obj;
            obj.x1 = x1_row[i];
            obj.y1 = y1_row[i];
            obj.x2 = x2_row[i];
            obj.y2 = y2_row[i];
            obj.prob = score;
            obj.label = (int)cls_row[i];
            proposals.push_back(obj);
            if ((int)proposals.size() >= config_.max_det)
                break;
        }

        objects.swap(proposals);
    }
    else
    {
        return false;
    }

    for (auto& obj : objects)
    {
        float x1 = obj.x1;
        float y1 = obj.y1;
        float x2 = obj.x2;
        float y2 = obj.y2;
        yolo26::scale_xyxy_inplace(x1, y1, x2, y2, img_w, img_h, lb, true);

        obj.x1 = x1;
        obj.y1 = y1;
        obj.x2 = x2;
        obj.y2 = y2;
    }

    // Apply NMS to remove duplicate detections
    objects = yolo26::nms(objects, config_.iou_threshold, config_.agnostic_nms);

    return true;
}
