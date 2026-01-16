import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch


def load_u8(path: Path, shape):
    arr = np.fromfile(path, dtype=np.uint8)
    return arr.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to yolo26_mask_parity binary")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    sys.path.insert(0, "/data/temp40/ultralytics")
    from ultralytics.utils import ops  # noqa: E402

    bin_path = Path(args.bin)
    if not bin_path.exists():
        raise SystemExit(f"Binary not found: {bin_path}")

    # Small shapes for fast parity check
    n = 7
    mask_dim = 32
    mh = 16
    mw = 16
    in_h = 64
    in_w = 64
    orig_h = 45
    orig_w = 80

    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            subprocess.check_call(
                [
                    str(bin_path),
                    str(n),
                    str(mask_dim),
                    str(mh),
                    str(mw),
                    str(in_h),
                    str(in_w),
                    str(orig_h),
                    str(orig_w),
                    str(seed),
                    str(td),
                ]
            )

            protos = np.fromfile(td / "protos.bin", dtype=np.float32).reshape(mask_dim, mh, mw)
            masks_in = np.fromfile(td / "masks_in.bin", dtype=np.float32).reshape(n, mask_dim)
            boxes_in = np.fromfile(td / "boxes_in.bin", dtype=np.float32).reshape(n, 4)
            boxes_orig = np.fromfile(td / "boxes_orig.bin", dtype=np.float32).reshape(n, 4)

            got_proc = load_u8(td / "masks_proc.bin", (n, in_h, in_w))
            got_scaled = load_u8(td / "masks_scaled.bin", (n, orig_h, orig_w))
            got_native = load_u8(td / "masks_native.bin", (n, orig_h, orig_w))

            protos_t = torch.from_numpy(protos)
            masks_in_t = torch.from_numpy(masks_in)
            boxes_in_t = torch.from_numpy(boxes_in)
            boxes_orig_t = torch.from_numpy(boxes_orig)

            ref_proc = ops.process_mask(protos_t, masks_in_t, boxes_in_t, (in_h, in_w), upsample=True).byte().numpy()
            ref_scaled = (ops.scale_masks(torch.from_numpy(ref_proc)[None].float(), (orig_h, orig_w))[0] > 0.5).byte().numpy()
            ref_native = ops.process_mask_native(protos_t, masks_in_t, boxes_orig_t, (orig_h, orig_w)).byte().numpy()

            if not np.array_equal(ref_proc, got_proc):
                raise SystemExit(f"seed={seed}: process_mask mismatch")
            if not np.array_equal(ref_scaled, got_scaled):
                raise SystemExit(f"seed={seed}: scale_masks mismatch")
            if not np.array_equal(ref_native, got_native):
                raise SystemExit(f"seed={seed}: process_mask_native mismatch")

    print("OK")


if __name__ == "__main__":
    main()

