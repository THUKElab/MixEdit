import sys
from typing import List


def postprocess(srcs: List[str], tgts: List[str], ids: List, max_len=10000):
    total = ids[-1] + 1 if isinstance(ids[0], int) else int(ids[-1].strip()) + 1
    results = ["" for _ in range(total)]
    for src, tgt, idx in zip(srcs, tgts, ids):
        src = src.replace("##", "").replace(" ", "")
        tgt = tgt.replace("##", "").replace(" ", "")
        if len(src) >= max_len or len(tgt) >= max_len:
            res = src
        else:
            res = tgt
        res = res.rstrip("\n")
        results[int(idx)] += res
    return results


if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        out_file = sys.argv[2]
        with open(input_file, "r") as f1:
            with open(out_file, "w") as f2:
                for line in f1:
                    line = line.strip().replace("##", "").replace(" ", "")
                    f2.write(line + "\n")

    elif len(sys.argv) == 6:
        input_file = sys.argv[1]
        cor_file = sys.argv[2]
        out_file = sys.argv[3]
        id_file = sys.argv[4]
        threshold = sys.argv[5]

        with open(input_file, "r") as f1, open(cor_file, "r") as f2, open(id_file, "r") as f3:
            src_lines, tgt_lines, id_lines = f1.readlines(), f2.readlines(), f3.readlines()
        post_results = postprocess(src_lines, tgt_lines, id_lines, max_len=int(threshold))

        with open(out_file, "w") as o:
            for post in post_results:
                o.write(post + "\n")
    else:
        raise ValueError(f"Invalid arguments: {len(sys.argv)}")
