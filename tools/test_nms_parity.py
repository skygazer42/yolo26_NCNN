import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def read_dets_txt(path: Path):
    out = []
    text = path.read_text().strip()
    if not text:
        return out
    for line in text.splitlines():
        x1, y1, x2, y2, s, c = line.strip().split()
        out.append(
            (
                float(np.float32(x1)),
                float(np.float32(y1)),
                float(np.float32(x2)),
                float(np.float32(y2)),
                float(np.float32(s)),
                int(c),
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to yolo26_nms_parity binary")
    ap.add_argument("--anchors", type=int, default=8400)
    ap.add_argument("--classes", type=int, default=80)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--agnostic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    sys.path.insert(0, "/data/temp40/ultralytics")
    from ultralytics.utils import nms  # noqa: E402

    bin_path = Path(args.bin)
    if not bin_path.exists():
        raise SystemExit(f"Binary not found: {bin_path}")

    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            subprocess.check_call(
                [
                    str(bin_path),
                    str(args.anchors),
                    str(args.classes),
                    str(args.max_det),
                    str(args.conf),
                    str(args.iou),
                    "1" if args.agnostic else "0",
                    str(seed),
                    str(td),
                ]
            )

            boxes = np.fromfile(td / "boxes_cxcywh.bin", dtype=np.float32).reshape(args.anchors, 4)
            scores = np.fromfile(td / "scores.bin", dtype=np.float32).reshape(args.anchors, args.classes)
            got = read_dets_txt(td / "dets.txt")

            boxes_t = torch.from_numpy(boxes).T  # (4, anchors)
            scores_t = torch.from_numpy(scores).T  # (nc, anchors)
            pred = torch.cat([boxes_t, scores_t], dim=0).unsqueeze(0)  # (1, 4+nc, anchors)

            ref = nms.non_max_suppression(
                pred,
                conf_thres=args.conf,
                iou_thres=args.iou,
                agnostic=args.agnostic,
                max_det=args.max_det,
                nc=args.classes,
                max_time_img=1e9,
            )[0]

            ref_list = [
                (
                    float(np.float32(x1)),
                    float(np.float32(y1)),
                    float(np.float32(x2)),
                    float(np.float32(y2)),
                    float(np.float32(sc)),
                    int(cls),
                )
                for x1, y1, x2, y2, sc, cls in ref.cpu().numpy().tolist()
            ]

            if len(ref_list) != len(got):
                raise SystemExit(f"seed={seed}: length mismatch: ref={len(ref_list)} got={len(got)}")

            def norm(d):
                x1, y1, x2, y2, s, c = d
                return (round(s, 6), int(c), round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))

            ref_norm = sorted(norm(d) for d in ref_list)
            got_norm = sorted(norm(d) for d in got)
            if ref_norm != got_norm:
                for i, (r, g) in enumerate(zip(ref_norm, got_norm)):
                    if r != g:
                        raise SystemExit(f"seed={seed}: mismatch at {i}: ref={r} got={g}")
                raise SystemExit(f"seed={seed}: mismatch (same prefix, different length?)")

    print("OK")


if __name__ == "__main__":
    main()
