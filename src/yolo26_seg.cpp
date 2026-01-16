#include "yolo26_seg.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "cpu.h"

#include "yolo26_seg_postprocess.h"
#include "yolo26_preprocess.h"
#include "yolo26_ops.h"
#include "yolo26_ncnn_mat.h"
#include "yolo26_topk.h"

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
    if (!yolo26::letterbox(bgr, config_.input_width, config_.input_height, 114, true, true, in_pad, lb))
        return false;
    yolo26::normalize_01_inplace(in_pad);

    const int pad_w = config_.input_width - lb.resized_w;
    const int pad_h = config_.input_height - lb.resized_h;

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
    objects.reserve(candidates.size());

    ncnn::Mat mask_feat = ncnn::Mat(config_.mask_dim, (int)candidates.size());
    for (size_t i = 0; i < candidates.size(); i++)
    {
        const Yolo26SegObject& src = candidates[i];
        Yolo26SegObject obj = src;

        float x1 = obj.x1;
        float y1 = obj.y1;
        float x2 = obj.x2;
        float y2 = obj.y2;
        yolo26::scale_xyxy_inplace(x1, y1, x2, y2, img_w, img_h, lb, true);

        obj.x1 = x1;
        obj.y1 = y1;
        obj.x2 = x2;
        obj.y2 = y2;

        float* mask_feat_ptr = mask_feat.row((int)i);
        std::copy(src.mask_feat.begin(), src.mask_feat.end(), mask_feat_ptr);

        objects.push_back(obj);
    }

    if (objects.empty())
        return true;

    ncnn::Mat mask_pred_result;
    ncnn::Mat masks;
    int proto_w = in_pad.w / 4;
    int proto_h = in_pad.h / 4;

    int proto_channels = proto.c;
    if (proto.dims == 4)
        proto_channels *= proto.d;
    if (proto_channels != config_.mask_dim)
        return false;

    ncnn::Mat proto_flat = proto;
    if (proto.dims == 3 || proto.dims == 4)
    {
        proto_w = proto.w;
        proto_h = proto.h;
        proto_flat = proto.reshape(proto_w * proto_h, proto_channels);
    }
    else if (proto.dims == 2)
    {
        if (proto.h != proto_channels)
            return false;
        if (proto_w * proto_h != proto.w)
            return false;
    }
    else
    {
        return false;
    }

    const int proto_scale_w = proto_w > 0 ? (in_pad.w / proto_w) : 0;
    const int proto_scale_h = proto_h > 0 ? (in_pad.h / proto_h) : 0;
    if (proto_scale_w <= 0 || proto_scale_h <= 0 || proto_scale_w != proto_scale_h)
        return false;
    const int proto_scale = proto_scale_w;

    yolo26_seg::matmul({mask_feat, proto_flat}, masks);
    yolo26_seg::sigmoid(masks);

    yolo26_seg::reshape(masks, masks, masks.h, proto_h, proto_w, 0);
    yolo26_seg::slice(masks, mask_pred_result, (pad_w / 2) / proto_scale, (in_pad.w - pad_w / 2) / proto_scale, 2);
    yolo26_seg::slice(mask_pred_result, mask_pred_result, (pad_h / 2) / proto_scale, (in_pad.h - pad_h / 2) / proto_scale, 1);
    yolo26_seg::interp(mask_pred_result, (float)proto_scale, img_w, img_h, mask_pred_result);

    for (size_t i = 0; i < objects.size(); i++)
    {
        objects[i].mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel((int)i));
        const int rx = (int)std::floor(objects[i].x1);
        const int ry = (int)std::floor(objects[i].y1);
        const int rw = (int)std::ceil(objects[i].x2 - objects[i].x1);
        const int rh = (int)std::ceil(objects[i].y2 - objects[i].y1);
        cv::Rect roi(rx, ry, rw, rh);
        roi &= cv::Rect(0, 0, img_w, img_h);
        if (roi.area() > 0)
            mask(roi).copyTo(objects[i].mask(roi));
    }

    return true;
}
