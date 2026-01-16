# YOLO26 + NCNN 部署

## 0. 环境准备

### 0.1 获取并编译安装 NCNN（示例路径：`/data/temp40/ncnn`）

```bash
git clone https://github.com/Tencent/ncnn.git /data/temp40/ncnn

cmake -S /data/temp40/ncnn -B /data/temp40/ncnn/build \
  -DNCNN_INSTALL_SDK=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/data/temp40/ncnn/install

cmake --build /data/temp40/ncnn/build -j
cmake --install /data/temp40/ncnn/build
```

本项目 CMake 需要的路径（示例）：
- `-Dncnn_DIR=/data/temp40/ncnn/install/lib/cmake/ncnn`

Vulkan（可选）：
- NCNN 编译选项：`-DNCNN_VULKAN=ON`
- 运行参数：`--gpu`

### 0.2 Python 导出依赖

本仓库的导出脚本使用：`ultralytics`、`torch`、`pnnx`。

```bash
python3 -m pip install -U ultralytics pnnx
```

`python/export_yolo26_end2end_raw_ncnn.py` 参数 `--ultralytics` 用于指定本地 Ultralytics 代码路径（默认：`/data/temp40/ultralytics`）。

## 1. 编译

依赖：
- CMake 3.10+
- OpenCV
- NCNN

```bash
cmake -S . -B build -Dncnn_DIR=/data/temp40/ncnn/install/lib/cmake/ncnn
cmake --build build -j
```

CMake 选项：
- `-DYOLO26_BUILD_SEG=ON|OFF`
- `-DYOLO26_BUILD_TOOLS=ON|OFF`

## 2. 可执行文件

- `build/yolo26_det`
- `build/yolo26_seg_demo`

## 3. 模型导出

生成物：
- 本仓库不提交：`*.pt`、`*_ncnn_model/`、`*_ncnn_e2e_raw_model/`、`*.onnx`
- 方案 A 导出目录：`<weights_stem>_ncnn_model/`
- 方案 B 导出目录：`<weights_stem>_ncnn_e2e_raw_model/`

### 3.1 方案 A：Ultralytics NCNN 导出（one2many）

Detection：
```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

Segmentation：
```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n-seg.pt").export(format="ncnn", imgsz=640, device="cpu")
PY
```

输出（imgsz=640）：
- detect：`out0` ≈ `(84, 8400)`，`[cx, cy, w, h] + 80 class scores`
- seg：`out0` ≈ `(116, 8400)`，`[cx, cy, w, h] + 80 class scores + 32 mask coeffs`
- seg：`out1` ≈ `(32, 160, 160)`（proto）

### 3.2 方案 B：end2end(one2one) RAW 导出（图内无 TopK）

Detection：
```bash
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n.pt --imgsz 640
```

Segmentation：
```bash
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n-seg.pt --imgsz 640
```

输出：
- detect：`out0` 为 `(anchors, 4+nc)`（box 为 `xyxy`）
- seg：`out0` 为 `(anchors, 4+nc+nm)`（box 为 `xyxy`），`out1` 为 proto

脚本参数：
- `--weights --imgsz --max-det --half --out-dir --ultralytics`

## 4. 运行

### 4.1 Detection

方案 A（NMS）：
```bash
./build/yolo26_det yolo26n_ncnn_model/model.ncnn.param yolo26n_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

方案 B（TopK）：
```bash
./build/yolo26_det yolo26n_ncnn_e2e_raw_model/model.ncnn.param yolo26n_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

### 4.2 Segmentation

方案 A（NMS）：
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_model/model.ncnn.param yolo26n-seg_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

方案 B（TopK）：
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.param yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

`yolo26_seg_demo` 额外参数：
- `--retina`

## 5. 参数

`yolo26_det` 默认值：
- `--conf 0.25 --iou 0.45 --max-det 300 --post auto --box cxcywh`

`yolo26_seg_demo` 默认值：
- `--conf 0.25 --iou 0.45 --max-det 300 --post auto --box cxcywh`

通用参数：
- `--conf --iou --max-det --post <auto|nms|topk> --box <cxcywh|xyxy> --agnostic --gpu`
- `--dedup`：TopK 后按 `--iou` 做一次 IoU 去重

## 6. 后处理匹配

- 方案 A 导出（one2many）：`--post=nms --box=cxcywh`
- 方案 B 导出（end2end RAW）：`--post=topk --box=xyxy`
- `--post=auto`：`--box=cxcywh` 时为 `nms`，`--box=xyxy` 时为 `topk`

## 7. 预处理与 IO 名称

预处理：
- BGR → RGB
- LetterBox：padding 值 `114`
- 归一化：`/255`

blob 名称：
- input：默认 `in0`，fallback：`images`、`data`
- output：默认 `out0`（seg proto 为 `out1`），fallback：`output0`/`output1`、`output`、`seg`

## 8. 对齐自检

```bash
python tools/run_parity.py --build-dir build
```

依赖：`build/yolo26_topk_parity`、`build/yolo26_nms_parity`、`build/yolo26_mask_parity`
