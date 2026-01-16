# yolo26_ncnn

Ultralytics YOLO26 推理示例（C++ + NCNN），支持 detection / segmentation。

- 部署/导出/运行：`docs/DEPLOYMENT.md`

## 编译

依赖：
- CMake 3.10+
- OpenCV
- NCNN

```bash
cmake -S . -B build -Dncnn_DIR=/data/temp40/ncnn/install/lib/cmake/ncnn
cmake --build build -j
```

## 模型导出

说明：
- 本仓库不提交：`*.pt`、`*_ncnn_model/`、`*_ncnn_e2e_raw_model/`、`*.onnx`

方案 A（Ultralytics NCNN 导出，one2many）：
```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="ncnn", imgsz=640, device="cpu")
YOLO("yolo26n-seg.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

方案 B（end2end(one2one) RAW 导出，图内无 TopK）：
```bash
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n.pt --imgsz 640
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n-seg.pt --imgsz 640
```

## 运行

Detection：
```bash
./build/yolo26_det yolo26n_ncnn_model/model.ncnn.param yolo26n_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
./build/yolo26_det yolo26n_ncnn_e2e_raw_model/model.ncnn.param yolo26n_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

Segmentation：
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_model/model.ncnn.param yolo26n-seg_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
./build/yolo26_seg_demo yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.param yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

## 参数

- `yolo26_det`：`--conf --iou --max-det --post --box --dedup --agnostic --gpu`
- `yolo26_seg_demo`：同上，额外 `--retina`
