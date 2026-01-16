import argparse
import sys
from pathlib import Path
from types import MethodType

import torch


def _patch_postprocess_identity(model) -> int:
    # Avoid torch.topk in end2end heads for NCNN export by making postprocess an identity op.
    # This keeps one2one (end2end) decoding, but outputs raw predictions (B, anchors, 4+nc) or (B, anchors, 4+nc+nm).
    def _identity_postprocess(self, preds):
        return preds

    patched = 0
    for m in model.modules():
        if hasattr(m, "postprocess") and getattr(m, "export", False) and getattr(m, "format", None) == "ncnn":
            m.postprocess = MethodType(_identity_postprocess, m)
            patched += 1
    return patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt", help="Path to YOLO26(.pt) weights (detect or seg)")
    ap.add_argument("--imgsz", type=int, default=640, help="Export image size")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections (only affects model attrs)")
    ap.add_argument("--half", action="store_true", help="Export FP16")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <weights_stem>_ncnn_e2e_raw_model)",
    )
    ap.add_argument(
        "--ultralytics",
        default="/data/temp40/ultralytics",
        help="Optional local Ultralytics checkout path for strict alignment",
    )
    args = ap.parse_args()

    ul_path = Path(args.ultralytics)
    if ul_path.exists():
        sys.path.insert(0, str(ul_path))

    from ultralytics import YOLO  # noqa: E402
    from ultralytics.nn.modules.head import Detect, Segment  # noqa: E402

    weights = Path(args.weights)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{weights.stem}_ncnn_e2e_raw_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    y = YOLO(str(weights))
    model = y.model

    model = model.fuse()
    model.eval()
    model.float()
    for p in model.parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, (Detect, Segment)):
            m.dynamic = False
            m.export = True
            m.format = "ncnn"
            m.max_det = args.max_det
            m.xyxy = False
            m.shape = None

    patched = _patch_postprocess_identity(model)
    if patched <= 0:
        raise SystemExit("No Detect/Segment modules patched; is this a YOLO26 end2end model?")

    im = torch.zeros(1, 3, args.imgsz, args.imgsz)

    import pnnx  # noqa: E402

    ncnn_args = dict(
        ncnnparam=(out_dir / "model.ncnn.param").as_posix(),
        ncnnbin=(out_dir / "model.ncnn.bin").as_posix(),
        ncnnpy=(out_dir / "model_ncnn.py").as_posix(),
    )
    pnnx_args = dict(
        ptpath=(out_dir / "model.pt").as_posix(),
        pnnxparam=(out_dir / "model.pnnx.param").as_posix(),
        pnnxbin=(out_dir / "model.pnnx.bin").as_posix(),
        pnnxpy=(out_dir / "model_pnnx.py").as_posix(),
        pnnxonnx=(out_dir / "model.pnnx.onnx").as_posix(),
    )

    # Some Ultralytics models can fail torch.jit trace check (graphs differ across invocations) even though the trace is
    # valid. Disable check_trace to make export robust.
    pnnx.export(model, inputs=im, **ncnn_args, **pnnx_args, fp16=args.half, device="cpu", check_trace=False)

    print(f"Saved: {out_dir}")
    print("Note: output is end2end one2one RAW (boxes are XYXY, no TopK in graph).")
    print("Use C++ with: --post=topk --box=xyxy (no NMS).")


if __name__ == "__main__":
    main()

