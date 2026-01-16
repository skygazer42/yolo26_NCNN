import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build", help="CMake build directory")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    build_dir = (root / args.build_dir).resolve()

    topk_bin = build_dir / "yolo26_topk_parity"
    nms_bin = build_dir / "yolo26_nms_parity"
    mask_bin = build_dir / "yolo26_mask_parity"

    subprocess.check_call(
        ["python", str(root / "tools/test_topk_parity.py"), "--bin", str(topk_bin), "--seeds", *map(str, args.seeds)]
    )
    subprocess.check_call(
        [
            "python",
            str(root / "tools/test_nms_parity.py"),
            "--bin",
            str(nms_bin),
            "--anchors",
            "2048",
            "--seeds",
            *map(str, args.seeds),
        ]
    )
    subprocess.check_call(
        ["python", str(root / "tools/test_mask_parity.py"), "--bin", str(mask_bin), "--seeds", *map(str, args.seeds)]
    )


if __name__ == "__main__":
    main()
