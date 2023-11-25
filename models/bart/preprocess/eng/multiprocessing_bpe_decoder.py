#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
        default="encoder.json"
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
        default="vocab.bpe"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        # default=["-"],
        default=["real/bea_train1/train.bpe.tgt"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        # default=["-"],
        default=["real/bea_train1/tgt.bart_bpe.tok"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        default=True,
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.decode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, line):
        # global bpe
        # return bpe.decode(tokens)
        global bpe
        tokens = line.split()
        len_1 = len(tokens)
        tokens_map = map(int, tokens)
        text = [bpe.decoder.get(token, token) for token in tokens_map]
        # print(text)
        for idx, tok in enumerate(text):
            if idx == 0:
                continue
            if tok[0] == "Ġ":
                tok = tok[1:]
                text[idx] = tok
            else:
                # assert text[idx - 1] != ""
                text[idx - 1] = text[idx - 1] + "@@"
        len_2 = len(text)
        assert len_1 == len_2, print(str(tokens), str(text))
        text = " ".join(text)
        # text = bytearray([bpe.byte_decoder[c] for c in text]).decode(
        #     "utf-8", errors=bpe.errors
        # )
        return text

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            res = self.decode(line.strip())
            dec_lines.append(res)
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
