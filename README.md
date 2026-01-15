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

## Model export

### Detection (end2end output, default Ultralytics export)

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="onnx", imgsz=640, opset=12, simplify=False, nms=False)
PY
```

Convert ONNX to NCNN with pnnx (recommended):

```bash
python -m pip install pnnx
pnnx yolo26n.onnx
```

This produces `yolo26n.ncnn.param` and `yolo26n.ncnn.bin`, with `output0` shape `1x300x6` and input name `images`.

### Detection (raw reg/cls outputs)

Some exports (for RKNN or custom pipelines) provide raw outputs:
`reg1/cls1`, `reg2/cls2`, `reg3/cls3` with shapes `1x4x80x80`, `1x80x80x80`, etc.
This demo will try raw outputs first and fall back to `output0`.

### Segmentation

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n-seg.pt").export(format="onnx", imgsz=640, opset=12, simplify=False, nms=False)
PY

python -m pip install pnnx
pnnx yolo26n-seg.onnx
```

The segmentation export uses:
- `output0` shape `1x300x38` (xyxy + score + class + 32 mask coeffs)
- `output1` shape `1x32x160x160` (mask prototypes)

## Run

Detection:
```bash
./build/yolo26_det yolo26n.ncnn.param yolo26n.ncnn.bin image.jpg output.jpg
```

Segmentation:
```bash
./build/yolo26_seg_demo yolo26n-seg.ncnn.param yolo26n-seg.ncnn.bin image.jpg output.jpg
```

## Notes

- Input blob name defaults to `data` for raw-output models; the code also tries `images`.
- Output blob names default to `reg1/cls1/...` for raw-output models and `output0` for end2end exports.
- If your model uses different blob names, update the config in code before calling `load()`.
