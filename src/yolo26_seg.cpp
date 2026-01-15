#include "yolo26_seg.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#include "ncnn/net.h"
#include "ncnn/cpu.h"

#include "yolo26_postprocess.h"
#include "yolo26_seg_postprocess.h"

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
    if (out_ret != 0 && config_.output_name != "output")
        out_ret = ex.extract("output", out);
    if (out_ret != 0 && config_.output_name != "output0")
        out_ret = ex.extract("output0", out);
    if (out_ret != 0)
        return false;

    int proto_ret = ex.extract(config_.proto_name.c_str(), proto);
    if (proto_ret != 0 && config_.proto_name != "seg")
        proto_ret = ex.extract("seg", proto);
    if (proto_ret != 0 && config_.proto_name != "output1")
        proto_ret = ex.extract("output1", proto);
    if (proto_ret != 0)
        return false;

    ncnn::Mat out_2d = out;
    if (out.dims == 3)
        out_2d = out.channel(0);

    std::vector<Yolo26SegObject> candidates;

    const int stride = 6 + config_.mask_dim;
    if (out_2d.w < stride)
        return false;

    const int num_dets = out_2d.h;
    for (int i = 0; i < num_dets; i++)
    {
        const float* p = out_2d.row(i);
        float score = p[4];
        if (score < config_.conf_threshold)
            continue;

        Yolo26SegObject obj;
        obj.rect.x = p[0];
        obj.rect.y = p[1];
        obj.rect.width = p[2] - p[0];
        obj.rect.height = p[3] - p[1];
        obj.prob = score;
        obj.label = (int)p[5];
        obj.mask_feat.resize(config_.mask_dim);
        std::copy(p + 6, p + stride, obj.mask_feat.begin());

        candidates.push_back(obj);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Yolo26SegObject& a, const Yolo26SegObject& b) { return a.prob > b.prob; });

    std::vector<Yolo26Object> nms_objects;
    nms_objects.reserve(candidates.size());
    for (const auto& obj : candidates)
    {
        Yolo26Object nms_obj;
        nms_obj.rect = obj.rect;
        nms_obj.label = obj.label;
        nms_obj.prob = obj.prob;
        nms_objects.push_back(nms_obj);
    }

    std::vector<int> picked;
    yolo26::nms_sorted_bboxes(nms_objects, picked, config_.nms_threshold);

    objects.clear();
    objects.reserve(picked.size());

    ncnn::Mat mask_feat = ncnn::Mat(config_.mask_dim, (int)picked.size());
    for (size_t i = 0; i < picked.size(); i++)
    {
        const Yolo26SegObject& src = candidates[picked[i]];
        Yolo26SegObject obj = src;

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

        float* mask_feat_ptr = mask_feat.row((int)i);
        std::copy(src.mask_feat.begin(), src.mask_feat.end(), mask_feat_ptr);

        objects.push_back(obj);
    }

    if (objects.empty())
        return true;

    ncnn::Mat mask_pred_result;
    ncnn::Mat masks;
    yolo26_seg::matmul({mask_feat, proto}, masks);
    yolo26_seg::sigmoid(masks);

    yolo26_seg::reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
    yolo26_seg::slice(masks, mask_pred_result, (pad_w / 2) / 4, (in_pad.w - pad_w / 2) / 4, 2);
    yolo26_seg::slice(mask_pred_result, mask_pred_result, (pad_h / 2) / 4, (in_pad.h - pad_h / 2) / 4, 1);
    yolo26_seg::interp(mask_pred_result, 4.0f, img_w, img_h, mask_pred_result);

    for (size_t i = 0; i < objects.size(); i++)
    {
        objects[i].mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel((int)i));
        cv::Rect roi = objects[i].rect;
        roi &= cv::Rect(0, 0, img_w, img_h);
        if (roi.area() > 0)
            mask(roi).copyTo(objects[i].mask(roi));
    }

    return true;
}
