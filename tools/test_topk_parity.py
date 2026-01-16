import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def torch_get_topk(scores: np.ndarray, max_det: int):
    assert scores.ndim == 2
    anchors, nc = scores.shape
    k = min(max_det, anchors)
    scores_t = torch.from_numpy(scores).unsqueeze(0)  # (1, anchors, nc)

    ori_index = scores_t.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)  # (1, k, 1)
    gathered = scores_t.gather(dim=1, index=ori_index.repeat(1, 1, nc))  # (1, k, nc)
    flat = gathered.flatten(1)  # (1, k*nc)
    top_scores, flat_index = flat.topk(k)  # (1, k)
    idx = ori_index[torch.arange(1)[..., None], flat_index // nc]  # (1, k, 1)
    cls = (flat_index % nc).to(torch.int64)  # (1, k)

    out = []
    for i in range(k):
        out.append((np.float32(top_scores[0, i].item()), int(cls[0, i].item()), int(idx[0, i, 0].item())))
    return out


def read_cpp_topk(path: Path):
    out = []
    for line in path.read_text().strip().splitlines():
        s, c, a = line.strip().split()
        out.append((np.float32(float(s)), int(c), int(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to yolo26_topk_parity binary")
    ap.add_argument("--anchors", type=int, default=8400)
    ap.add_argument("--classes", type=int, default=80)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        raise SystemExit(f"Binary not found: {bin_path}")

    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            scores_bin = td / "scores.bin"
            topk_txt = td / "topk.txt"
            subprocess.check_call(
                [
                    str(bin_path),
                    str(args.anchors),
                    str(args.classes),
                    str(args.max_det),
                    str(seed),
                    str(scores_bin),
                    str(topk_txt),
                ]
            )

            scores = np.fromfile(scores_bin, dtype=np.float32).reshape(args.anchors, args.classes)
            ref = torch_get_topk(scores, args.max_det)
            got = read_cpp_topk(topk_txt)

            if len(ref) != len(got):
                raise SystemExit(f"seed={seed}: length mismatch: ref={len(ref)} got={len(got)}")

            i = 0
            while i < len(ref):
                rs = ref[i][0]
                j = i
                while j < len(ref) and ref[j][0] == rs:
                    j += 1
                k = i
                while k < len(got) and got[k][0] == rs:
                    k += 1

                ref_group = sorted((c, a) for _, c, a in ref[i:j])
                got_group = sorted((c, a) for _, c, a in got[i:k])
                if ref_group != got_group:
                    raise SystemExit(
                        f"seed={seed} score={float(rs):.9f}: group mismatch "
                        f"expected={ref_group} got={got_group}"
                    )

                if (j - i) != (k - i):
                    raise SystemExit(
                        f"seed={seed} score={float(rs):.9f}: group size mismatch "
                        f"expected={j - i} got={k - i}"
                    )

                i = j

    print("OK")


if __name__ == "__main__":
    main()
