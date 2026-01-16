# yolo26_ncnn

Ultralytics YOLO26 detection/segmentation demo in **C++ + NCNN**.

- 中文部署/对齐说明：`docs/DEPLOYMENT.md`

## Quickstart

### Build

Prerequisites:
- CMake 3.10+
- OpenCV
- NCNN

```bash
cmake -S . -B build -Dncnn_DIR=/data/temp40/ncnn/install/lib/cmake/ncnn
cmake --build build -j
```

If `find_package(ncnn)` fails, pass `-Dncnn_DIR=<path>/lib/cmake/ncnn`.

### Choose postprocess mode (important)

This repo supports **two aligned pipelines**:

- **Mode A (recommended for “official NCNN export alignment”)**: one2many output from `YOLO(...).export(format="ncnn")` + **C++ NMS**
  - Typical output: detect `84x8400`, seg `116x8400`
  - Box format: `cxcywh`
  - Run with: `--post=nms --box=cxcywh`
- **Mode B (recommended for “YOLO26 end2end / NMS-free semantics”)**: end2end(one2one) **RAW** export (no `torch.topk` in graph) + **C++ TopK**
  - Box format: `xyxy`
  - Run with: `--post=topk --box=xyxy` (no NMS)

`--post=auto` picks `nms` for `--box=cxcywh`, and `topk` for `--box=xyxy` (be explicit if you’re debugging).

## Model export

Python export prerequisites:
- `ultralytics` (YOLO26 support) + `torch`
- `pnnx` Python module (comes with NCNN builds)

The end2end RAW export scripts will prefer a local Ultralytics checkout at `/data/temp40/ultralytics` (for strict alignment) if it exists.

### Detection

#### Mode A: official Ultralytics NCNN export (one2many)

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

Run:
```bash
./build/yolo26_det yolo26n_ncnn_model/model.ncnn.param yolo26n_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

#### Mode B: end2end(one2one) RAW export for NCNN (NMS-free)

Ultralytics disables end2end `torch.topk` for NCNN export. These scripts keep the one2one head but remove TopK from the graph, so you can run TopK in C++:

```bash
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n.pt --imgsz 640
```

Run:
```bash
./build/yolo26_det yolo26n_ncnn_e2e_raw_model/model.ncnn.param yolo26n_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

### Segmentation

#### Mode A: official Ultralytics NCNN export (one2many)

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n-seg.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

Run:
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_model/model.ncnn.param yolo26n-seg_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

#### Mode B: end2end(one2one) RAW export for NCNN (NMS-free)

```bash
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n-seg.pt --imgsz 640
```

Run:
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.param yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

## CLI options

Usage:
- `./build/yolo26_det <param> <bin> <image> [output] [options]` (default output: `yolo26_result.jpg`)
- `./build/yolo26_seg_demo <param> <bin> <image> [output] [options]` (default output: `yolo26_seg_result.jpg`)

`yolo26_det`:
- `--conf <float>` (default `0.25`)
- `--iou <float>` (NMS IoU, default `0.45`)
- `--max-det <int>` (default `300`)
- `--post <auto|nms|topk>` (default `auto`)
- `--box <cxcywh|xyxy>` (default `cxcywh`)
- `--dedup` (apply IoU de-dup after TopK, uses `--iou`)
- `--agnostic` (class-agnostic NMS)
- `--gpu` (Vulkan, if NCNN built with it)

`yolo26_seg_demo` (same flags, different defaults):
- `--conf <float>` (default `0.5`)
- `--iou <float>` (NMS IoU, default `0.45`)
- `--max-det <int>` (default `300`)
- `--post <auto|nms|topk>` (default `auto`)
- `--box <cxcywh|xyxy>` (default `cxcywh`)
- `--dedup` (apply IoU de-dup after TopK, uses `--iou`)
- `--agnostic` (class-agnostic NMS)
- `--retina` (retina masks path)
- `--gpu` (Vulkan, if NCNN built with it)

## Troubleshooting

- **Lots of duplicate boxes**: you are likely running one2many outputs (Mode A) with `--post=topk`. Use `--post=nms --box=cxcywh` for official exports.
- **Too many “garbage boxes” with TopK**: ensure you’re using Mode B (end2end RAW) and `--box=xyxy`; then raise `--conf` (e.g. `0.4/0.5`) or reduce `--max-det` (e.g. `100`). If you only want cleaner visualization, use Mode A (NMS).
- **Want TopK but cleaner boxes**: add `--dedup` (runs a fast IoU de-dup after TopK on at most `--max-det` boxes).
- **Blob names mismatch**: input defaults to `in0` (also tries `images`/`data`), output defaults to `out0`/`out1` (also tries `output0`/`output1`). If your model uses different names, update config in code before `load()`.

## Parity checks (optional)

This repo includes small TopK/NMS/mask parity tests:
```bash
python tools/run_parity.py --build-dir build
```

## Notes

- Preprocess matches Ultralytics `LetterBox`: RGB input, padding value `114`, `/255` normalization.
