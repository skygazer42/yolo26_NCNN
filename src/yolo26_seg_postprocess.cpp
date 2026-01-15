#include "yolo26_seg_postprocess.h"

#include "ncnn/layer.h"

namespace yolo26_seg {

void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    ncnn::ParamDict pd;
    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);
    pd.set(10, ends);
    pd.set(11, axes);

    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);

    delete op;
}

void interp(const ncnn::Mat& in, float scale, int out_w, int out_h, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    ncnn::ParamDict pd;
    pd.set(0, 2);
    pd.set(1, scale);
    pd.set(2, scale);
    pd.set(3, out_h);
    pd.set(4, out_w);

    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);

    delete op;
}

void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    ncnn::ParamDict pd;
    pd.set(0, w);
    pd.set(1, h);
    if (d > 0)
        pd.set(11, d);
    pd.set(2, c);

    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);

    delete op;
}

void sigmoid(ncnn::Mat& bottom)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Sigmoid");
    op->create_pipeline(opt);
    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}

void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("MatMul");

    ncnn::ParamDict pd;
    pd.set(0, 0);

    op->load_param(pd);
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);
    delete op;
}

}  // namespace yolo26_seg
