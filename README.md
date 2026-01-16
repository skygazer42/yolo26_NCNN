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

### Detection (Ultralytics NCNN export)

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

This creates `yolo26n_ncnn_model/` containing `model.ncnn.param` and `model.ncnn.bin`.
The model output `out0` shape is `84x8400`: `[cx, cy, w, h] + 80 class scores` (NMS-free; this demo applies TopK).

### Segmentation

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n-seg.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

The segmentation export uses:
- `out0` shape `116x8400` ([cx, cy, w, h] + 80 class scores + 32 mask coeffs, NMS-free TopK in this demo)
- `out1` shape `32x160x160` (mask prototypes)

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

- YOLO26 is end-to-end NMS-free; Ultralytics NCNN export disables the end2end branch (TopK unsupported in NCNN), so this demo runs the same TopK selection in C++ (no NMS).
- Preprocess matches Ultralytics `LetterBox`: RGB input, padding value `114`, `/255` normalization.
- Input blob name defaults to `in0`; the code also tries `images` and `data`.
- Output blob names default to `out0` (and `out1` for segmentation); the code also tries `output0`/`output1`.
- If your model uses different blob names, update the config in code before calling `load()`.
