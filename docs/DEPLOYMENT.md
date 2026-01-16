# 部署与对齐说明（YOLO26 + NCNN + C++）

本项目支持两种推理/后处理路径：

- **方案 A：官方 Ultralytics `export(format="ncnn")`（one2many 输出）+ C++ NMS**
- **方案 B：YOLO26 end2end(one2one) RAW 导出（图里无 TopK）+ C++ TopK（NMS-free）**

重点：YOLO26 “NMS-free”指的是 **end2end(one2one)+TopK** 的语义；但官方 NCNN 导出会因为 NCNN 图不支持 `torch.topk` 而退化为 one2many dense 输出，所以需要 NMS 才能去重并对齐官方 NCNN 后端行为。

---

## 1. 依赖与编译

依赖：
- CMake 3.10+
- OpenCV
- NCNN

编译：
```bash
cmake -S . -B build -Dncnn_DIR=/data/temp40/ncnn/install/lib/cmake/ncnn
cmake --build build -j
```

---

## 2. 选择哪种方案？

### 方案 A：one2many + NMS（推荐用于“对齐官方 NCNN 导出/结果”）

适用场景：
- 你用的是 `YOLO(...).export(format="ncnn")` 的默认导出目录 `*_ncnn_model/`
- 你想与 **Ultralytics 的 NCNN 后端**输出行为一致（通常需要 NMS）

特点：
- 输出通常是 `84x8400`（detect）或 `116x8400`（seg）
- 原始 box 通常是 **CXCYWH**
- 后处理用 **NMS** 去重会更“干净”

### 方案 B：end2end RAW + TopK（推荐用于“拿到 YOLO26 end2end/NMS-free 语义”）

适用场景：
- 你希望保持 YOLO26 的 end2end(one2one) 语义（不走 NMS）
- 你接受把 TopK 放在 C++ 侧（因为 NCNN 图内不支持 `torch.topk`）

特点：
- 输出是 **one2one RAW**：`(anchors, 4+nc)` 或 `((anchors, 4+nc+nm), proto)`（没有 `300x6`）
- box 通常是 **XYXY**
- C++ 做 **TopK**（不做 NMS）

---

## 3. 导出模型

### 3.1 方案 A：官方 Ultralytics NCNN 导出

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

会生成 `yolo26n_ncnn_model/`、`yolo26n-seg_ncnn_model/`。

### 3.2 方案 B：end2end RAW（one2one）导出（绕过图内 TopK）

Detection：
```bash
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n.pt --imgsz 640
```

Segmentation：
```bash
python python/export_yolo26_end2end_raw_ncnn.py --weights yolo26n-seg.pt --imgsz 640
```

会生成 `*_ncnn_e2e_raw_model/`。

---

## 4. 运行（C++）

### 4.1 Detection

方案 A（NMS）：
```bash
./build/yolo26_det yolo26n_ncnn_model/model.ncnn.param yolo26n_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

方案 B（TopK，NMS-free）：
```bash
./build/yolo26_det yolo26n_ncnn_e2e_raw_model/model.ncnn.param yolo26n_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

### 4.2 Segmentation

方案 A（NMS）：
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_model/model.ncnn.param yolo26n-seg_ncnn_model/model.ncnn.bin image.jpg out.jpg --post=nms --box=cxcywh
```

方案 B（TopK，NMS-free）：
```bash
./build/yolo26_seg_demo yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.param yolo26n-seg_ncnn_e2e_raw_model/model.ncnn.bin image.jpg out.jpg --post=topk --box=xyxy
```

---

## 5. TopK 看起来“不如 NMS 干净”的原因与建议

TopK-only 不做重叠抑制，因此视觉上经常会比 NMS “更乱”。这在以下情况尤为明显：

1) **你其实在跑 one2many 输出，但用了 `--post=topk`**  
   - 现象：大量重复框/贴边框  
   - 处理：改用 `--post=nms --box=cxcywh`

2) **box 格式用错（`cxcywh`/`xyxy`）**  
   - 现象：框位置/大小明显异常（“垃圾框”）  
   - 处理：方案 A 用 `--box=cxcywh`，方案 B 用 `--box=xyxy`

3) **阈值偏低**（TopK 会把全图最高分的候选都拉进来）  
   - 建议：提高 `--conf`（例如 0.4/0.5），或减小 `--max-det`（例如 100）

4) **想保留 TopK 但视觉更“干净”**  
   - 可加 `--dedup`：TopK 之后对最多 `--max-det` 个候选做一次 IoU 去重（使用 `--iou` 阈值）。

如果你只追求“更干净的可视化”，不严格要求 end2end 语义，也可以直接使用 NMS 方案（A）。

---

## 6. 一致性自检（可选）

本项目带了 TopK/NMS/Mask 的对齐测试：
```bash
python tools/run_parity.py --build-dir build
```
