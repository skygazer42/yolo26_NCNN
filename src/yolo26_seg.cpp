#include "yolo26_seg.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "cpu.h"

#include "yolo26_preprocess.h"
#include "yolo26_ops.h"
#include "yolo26_ncnn_mat.h"
#include "yolo26_topk.h"
#include "yolo26_mask.h"

Yolo26Seg::Yolo26Seg(const Yolo26SegConfig& config)
    : config_(config), net_(std::make_shared<ncnn::Net>())
{
}

Yolo26Seg::~Yolo26Seg() = default;

bool Yolo26Seg::load(const std::string& param_path, const std::string& bin_path)
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

bool Yolo26Seg::detect(const cv::Mat& bgr, std::vector<Yolo26SegObject>& objects) const
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
    ncnn::Mat proto;
    int out_ret = ex.extract(config_.output_name.c_str(), out);
    if (out_ret != 0 && config_.output_name != "out0")
        out_ret = ex.extract("out0", out);
    if (out_ret != 0 && config_.output_name != "output0")
        out_ret = ex.extract("output0", out);
    if (out_ret != 0 && config_.output_name != "output")
        out_ret = ex.extract("output", out);
    if (out_ret != 0)
        return false;

    int proto_ret = ex.extract(config_.proto_name.c_str(), proto);
    if (proto_ret != 0 && config_.proto_name != "out1")
        proto_ret = ex.extract("out1", proto);
    if (proto_ret != 0 && config_.proto_name != "seg")
        proto_ret = ex.extract("seg", proto);
    if (proto_ret != 0 && config_.proto_name != "output1")
        proto_ret = ex.extract("output1", proto);
    if (proto_ret != 0)
        return false;

    ncnn::Mat out_2d;
    if (!yolo26::to_mat2d(out, out_2d))
        return false;

    std::vector<Yolo26SegObject> candidates;

    const int det_dim = 4 + config_.num_classes;
    const int det_mask_dim = det_dim + config_.mask_dim;
    const int row_stride = 6 + config_.mask_dim;

    // Ultralytics NCNN export (end2end disabled) typically outputs [4+nc+nm, num_anchors] i.e. (116, 8400).
    if (out_2d.h == det_mask_dim && out_2d.w > 0)
    {
        const int num_anchors = out_2d.w;

        const float* box_cx = out_2d.row(0);
        const float* box_cy = out_2d.row(1);
        const float* box_w = out_2d.row(2);
        const float* box_h = out_2d.row(3);

        std::vector<const float*> score_rows(config_.num_classes);
        for (int c = 0; c < config_.num_classes; c++)
            score_rows[c] = out_2d.row(4 + c);

        std::vector<const float*> mask_rows(config_.mask_dim);
        for (int m = 0; m < config_.mask_dim; m++)
            mask_rows[m] = out_2d.row(4 + config_.num_classes + m);

        const auto topk = yolo26::get_topk_index(
            num_anchors, config_.num_classes, config_.max_det,
            [&](int anchor, int cls) { return score_rows[cls][anchor]; });

        candidates.reserve(topk.size());
        for (const auto& cand : topk)
        {
            if (cand.score < config_.conf_threshold)
                continue;

            const int i = cand.anchor;
            const float cx = box_cx[i];
            const float cy = box_cy[i];
            const float w = box_w[i];
            const float h = box_h[i];

            Yolo26SegObject obj;
            obj.x1 = cx - w * 0.5f;
            obj.y1 = cy - h * 0.5f;
            obj.x2 = cx + w * 0.5f;
            obj.y2 = cy + h * 0.5f;
            obj.prob = cand.score;
            obj.label = cand.cls;
            obj.mask_feat.resize(config_.mask_dim);
            for (int m = 0; m < config_.mask_dim; m++)
                obj.mask_feat[m] = mask_rows[m][i];

            candidates.push_back(std::move(obj));
        }
    }
    // Some converters may output [num_anchors, 4+nc+nm] i.e. (8400, 116).
    else if (out_2d.w == det_mask_dim && out_2d.h > 0)
    {
        const int num_anchors = out_2d.h;
        const auto topk = yolo26::get_topk_index(
            num_anchors, config_.num_classes, config_.max_det,
            [&](int anchor, int cls) { return out_2d.row(anchor)[4 + cls]; });

        candidates.reserve(topk.size());
        for (const auto& cand : topk)
        {
            if (cand.score < config_.conf_threshold)
                continue;

            const float* p = out_2d.row(cand.anchor);
            const float cx = p[0];
            const float cy = p[1];
            const float w = p[2];
            const float h = p[3];

            Yolo26SegObject obj;
            obj.x1 = cx - w * 0.5f;
            obj.y1 = cy - h * 0.5f;
            obj.x2 = cx + w * 0.5f;
            obj.y2 = cy + h * 0.5f;
            obj.prob = cand.score;
            obj.label = cand.cls;
            obj.mask_feat.resize(config_.mask_dim);
            const float* mask_ptr = p + 4 + config_.num_classes;
            std::copy(mask_ptr, mask_ptr + config_.mask_dim, obj.mask_feat.begin());

            candidates.push_back(std::move(obj));
        }
    }
    // End-to-end export outputs (already top-k): [num_dets, 6+nm] i.e. (300, 38) with xyxy + score + cls + mask coeffs.
    else if (out_2d.w == row_stride && out_2d.h > 0)
    {
        const int num_dets = out_2d.h;
        candidates.reserve(std::min(num_dets, config_.max_det));
        for (int i = 0; i < num_dets; i++)
        {
            const float* p = out_2d.row(i);
            float score = p[4];
            if (score < config_.conf_threshold)
                continue;

            Yolo26SegObject obj;
            obj.x1 = p[0];
            obj.y1 = p[1];
            obj.x2 = p[2];
            obj.y2 = p[3];
            obj.prob = score;
            obj.label = (int)p[5];
            obj.mask_feat.resize(config_.mask_dim);
            std::copy(p + 6, p + row_stride, obj.mask_feat.begin());

            candidates.push_back(std::move(obj));
            if ((int)candidates.size() >= config_.max_det)
                break;
        }
    }
    // Some converters may output [6+nm, num_dets] i.e. (38, 300).
    else if (out_2d.h == row_stride && out_2d.w > 0)
    {
        const int num_dets = out_2d.w;
        const float* x1_row = out_2d.row(0);
        const float* y1_row = out_2d.row(1);
        const float* x2_row = out_2d.row(2);
        const float* y2_row = out_2d.row(3);
        const float* score_row = out_2d.row(4);
        const float* cls_row = out_2d.row(5);

        std::vector<const float*> mask_rows(config_.mask_dim);
        for (int m = 0; m < config_.mask_dim; m++)
            mask_rows[m] = out_2d.row(6 + m);

        candidates.reserve(std::min(num_dets, config_.max_det));
        for (int i = 0; i < num_dets; i++)
        {
            const float score = score_row[i];
            if (score < config_.conf_threshold)
                continue;

            Yolo26SegObject obj;
            obj.x1 = x1_row[i];
            obj.y1 = y1_row[i];
            obj.x2 = x2_row[i];
            obj.y2 = y2_row[i];
            obj.prob = score;
            obj.label = (int)cls_row[i];
            obj.mask_feat.resize(config_.mask_dim);
            for (int m = 0; m < config_.mask_dim; m++)
                obj.mask_feat[m] = mask_rows[m][i];

            candidates.push_back(std::move(obj));
            if ((int)candidates.size() >= config_.max_det)
                break;
        }
    }
    else
    {
        return false;
    }

    objects.clear();
    if (candidates.empty())
        return true;

    const int n = (int)candidates.size();
    ncnn::Mat mask_feat = ncnn::Mat(config_.mask_dim, n);
    std::vector<yolo26::BoxXYXY> boxes_input;
    boxes_input.reserve(n);
    for (int i = 0; i < n; i++)
    {
        const Yolo26SegObject& src = candidates[i];
        yolo26::BoxXYXY box;
        box.x1 = src.x1;
        box.y1 = src.y1;
        box.x2 = src.x2;
        box.y2 = src.y2;
        boxes_input.push_back(box);

        float* mask_feat_ptr = mask_feat.row(i);
        std::copy(src.mask_feat.begin(), src.mask_feat.end(), mask_feat_ptr);
    }

    // Normalize proto to CHW layout (mask_dim, mh, mw)
    ncnn::Mat proto_chw = proto;
    if (proto.dims == 4)
    {
        proto_chw = proto.reshape(proto.w, proto.h, proto.c * proto.d);
    }
    else if (proto.dims == 2)
    {
        if (proto.h != config_.mask_dim)
            return false;
        const int side = (int)std::round(std::sqrt((double)proto.w));
        if (side <= 0 || side * side != proto.w)
            return false;
        proto_chw = proto.reshape(side, side, proto.h);
    }

    if (proto_chw.dims != 3 || proto_chw.c != config_.mask_dim)
        return false;

    const bool retina_masks = config_.retina_masks;

    if (retina_masks)
    {
        std::vector<yolo26::BoxXYXY> boxes_orig;
        boxes_orig.reserve(n);
        for (const auto& b : boxes_input)
        {
            float x1 = b.x1;
            float y1 = b.y1;
            float x2 = b.x2;
            float y2 = b.y2;
            yolo26::scale_xyxy_inplace(x1, y1, x2, y2, img_w, img_h, lb, true);
            yolo26::BoxXYXY box;
            box.x1 = x1;
            box.y1 = y1;
            box.x2 = x2;
            box.y2 = y2;
            boxes_orig.push_back(box);
        }

        std::vector<cv::Mat> masks_orig;
        if (!yolo26::process_mask_native(proto_chw, mask_feat, boxes_orig, img_h, img_w, masks_orig))
            return false;

        objects.reserve((size_t)masks_orig.size());
        for (int i = 0; i < n; i++)
        {
            if (cv::countNonZero(masks_orig[i]) <= 0)
                continue;
            Yolo26SegObject obj = candidates[i];
            obj.x1 = boxes_orig[i].x1;
            obj.y1 = boxes_orig[i].y1;
            obj.x2 = boxes_orig[i].x2;
            obj.y2 = boxes_orig[i].y2;
            obj.mask = masks_orig[i];
            objects.push_back(std::move(obj));
        }
        return true;
    }

    std::vector<cv::Mat> masks_input;
    if (!yolo26::process_mask(
            proto_chw, mask_feat, boxes_input, config_.input_height, config_.input_width, true, masks_input))
        return false;

    std::vector<int> keep;
    keep.reserve(n);
    std::vector<cv::Mat> masks_input_keep;
    masks_input_keep.reserve((size_t)n);
    for (int i = 0; i < n; i++)
    {
        if (cv::countNonZero(masks_input[i]) <= 0)
            continue;
        keep.push_back(i);
        masks_input_keep.push_back(masks_input[i]);
    }

    std::vector<cv::Mat> masks_orig;
    if (!yolo26::scale_masks(masks_input_keep, img_h, img_w, masks_orig, true))
        return false;

    objects.reserve(keep.size());
    for (size_t j = 0; j < keep.size(); j++)
    {
        const int i = keep[j];
        Yolo26SegObject obj = candidates[i];

        float x1 = obj.x1;
        float y1 = obj.y1;
        float x2 = obj.x2;
        float y2 = obj.y2;
        yolo26::scale_xyxy_inplace(x1, y1, x2, y2, img_w, img_h, lb, true);
        obj.x1 = x1;
        obj.y1 = y1;
        obj.x2 = x2;
        obj.y2 = y2;

        obj.mask = masks_orig[j];
        objects.push_back(std::move(obj));
    }

    return true;
}
