# yolo26_ncnn

NCNN C++ demo for Ultralytics YOLO26 detection and segmentation, based on the YOLO26 output layouts and NCNN usage
patterns from the referenced repos.

## Build

Prerequisites:
- CMake 3.10+
- OpenCV
- NCNN

```bash
cmake -S . -B build
cmake --build build -j
```

If `find_package(ncnn)` fails, pass `-Dncnn_DIR=<path>/lib/cmake/ncnn` (e.g. `-Dncnn_DIR=/data/temp40/ncnn/install/lib/cmake/ncnn`).

## Model export

### Detection

#### A) Official Ultralytics NCNN export (one2many, needs NMS)

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

This creates `yolo26n_ncnn_model/` containing `model.ncnn.param` and `model.ncnn.bin`.
The model output `out0` shape is typically `84x8400`: `[cx, cy, w, h] + 80 class scores` (one2many output, requires NMS).

Run:
```bash
./build/yolo26_det yolo26n_ncnn_model/model.ncnn.param yolo26n_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

#### B) YOLO26 end2end-style export for NCNN (one2one RAW, NMS-free)

Ultralytics disables end2end TopK for NCNN export. To keep the end2end one2one head while removing the `torch.topk`
from the graph, export a "raw" end2end model and run TopK in C++:

```bash
python python/export_yolo26_det_end2end_raw_ncnn.py --weights yolo26n.pt --imgsz 640
```

Run:
```bash
./build/yolo26_det yolo26n_ncnn_e2e_raw_model/model.ncnn.param yolo26n_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

### Segmentation

#### A) Official Ultralytics NCNN export (one2many, needs NMS)

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n-seg.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

The segmentation export uses:
- `out0` shape `116x8400` ([cx, cy, w, h] + 80 class scores + 32 mask coeffs, one2many output, requires NMS)
- `out1` shape `32x160x160` (mask prototypes)

Run:
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_model/model.ncnn.param yolo26n-seg_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

#### B) YOLO26 end2end-style export for NCNN (one2one RAW, NMS-free)

```bash
python python/export_yolo26_seg_end2end_raw_ncnn.py --weights yolo26n-seg.pt --imgsz 640
```

Run:
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.param yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

## Run

Detection:
```bash
./build/yolo26_det yolo26n_ncnn_model/model.ncnn.param yolo26n_ncnn_model/model.ncnn.bin image.jpg output.jpg
```

Segmentation:
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_model/model.ncnn.param yolo26n-seg_ncnn_model/model.ncnn.bin image.jpg output.jpg
```

## Notes

- YOLO26 end2end is NMS-free only when using the one2one head + TopK. Ultralytics disables end2end TopK for NCNN export, so official NCNN export outputs one2many predictions that require NMS.
- Preprocess matches Ultralytics `LetterBox`: RGB input, padding value `114`, `/255` normalization.
- Input blob name defaults to `in0`; the code also tries `images` and `data`.
- Output blob names default to `out0` (and `out1` for segmentation); the code also tries `output0`/`output1`.
- If your model uses different blob names, update the config in code before calling `load()`.
