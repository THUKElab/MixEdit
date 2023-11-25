import os
import sys
import argparse
from bert import tokenization
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--lowercase', default=False, type=bool)
args = parser.parse_args()

tokenizer = tokenization.FullTokenizer(
    # vocab_file=os.path.join(os.path.dirname(__file__), "vocab_v1.txt"),
    vocab_file=os.path.join(os.path.dirname(__file__), "vocab_v2.txt"),
    do_lower_case=args.lowercase,  # Set to True to avoid most [UNK]
)


def split(line):
    line = line.strip()
    # 2023.09.24 注释下行，避免连续单词间空格丢失
    # line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        return ''
    tokens = tokenizer.tokenize(line)
    return ' '.join(tokens)


with Pool(64) as pool:
    for ret in pool.imap(split, tqdm(sys.stdin), chunksize=1024):
        print(ret)
