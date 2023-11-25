import argparse
import os
import sys
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# English resources
SPACY_MODEL = spacy.load('en_core_web_sm')


def tokenize_batch(text_list: List[str], no_space: bool = False):
    if no_space:
        text_list = [remove_space(x) for x in text_list]
    docs = SPACY_MODEL.pipe(
        text_list,
        batch_size=1024,
        disable=['parser', 'tagger', 'ner'],
    )
    docs = [[x.text for x in line] for line in docs]
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_input", type=str)
    parser.add_argument("--dir_output", type=str)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--lang", type=str, default="eng", choices=["eng", "zho"])
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    print(args)

    os.makedirs(args.dir_output, exist_ok=True)

    if args.lang == "eng":
        forward_model_name = "Helsinki-NLP/opus-mt-en-zh"
        backward_model_name = "Helsinki-NLP/opus-mt-zh-en"
    elif args.lang == "zho":
        forward_model_name = "Helsinki-NLP/opus-mt-zh-en"
        backward_model_name = "Helsinki-NLP/opus-mt-en-zh"
    else:
        raise ValueError
    print(f"Load forward_model_name: {forward_model_name}")
    print(f"Load backward_model_name: {backward_model_name}")

    # Build forward model
    forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
    forward_model = MarianMTModel.from_pretrained(forward_model_name, from_tf=False)
    forward_model.to(args.device)
    print(forward_tokenizer.supported_language_codes)
    print(forward_model.config)
    print(forward_model)

    # Build backward model
    backward_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)
    backward_model = MarianMTModel.from_pretrained(backward_model_name, from_tf=False)
    backward_model.to(args.device)
    print(backward_tokenizer.supported_language_codes)
    print(backward_model.config)
    print(backward_model)

    text = []
    with open(args.file_input, "r") as f:
        for line in f:
            line = line.strip()
            if args.lang == "zho":
                line = line.replace(" ", "")
                if len(line) > args.max_len:
                    print(f"Too long sentence, truncate to {args.max_len} : {line}")
                    line = line[:args.max_len]
            elif len(line.split()) > args.max_len:
                print(f"Too long sentence, truncate to {args.max_len}: {line}")
                line = " ".join(line.split()[:args.max_len])
            text.append(line)

    with open(f"{args.dir_output}/src.txt", "w", encoding="utf-8") as f1, \
            open(f"{args.dir_output}/tgt.txt", "w", encoding="utf-8") as f2, \
            open(f"{args.dir_output}/bridge.txt", "w", encoding="utf-8") as f3:
        for idx in tqdm(range(0, len(text), args.bsz)):
            batch = text[idx: idx + args.bsz]

            forward_translated = forward_model.generate(
                **forward_tokenizer(batch, return_tensors="pt", padding=True).to(args.device),
                max_new_tokens=128,
            )
            forward_output = [
                forward_tokenizer.decode(
                    t, skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for t in forward_translated
            ]

            backward_translated = backward_model.generate(
                **backward_tokenizer(forward_output, return_tensors="pt", padding=True).to(args.device),
                max_new_tokens=128,
            )
            backward_output = [
                backward_tokenizer.decode(
                    t, skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for t in backward_translated
            ]

            if args.lang == "eng":
                dek_backward_output = tokenize_batch(backward_output)
                for line in dek_backward_output:
                    line = " ".join(line)
                    f1.write(line.strip() + "\n")
            else:
                for line in backward_output:
                    f1.write(line.strip() + "\n")

            for line in batch:
                f2.write(line.strip() + "\n")

            for line in forward_output:
                f3.write(line.strip() + "\n")


if __name__ == '__main__':
    main()
