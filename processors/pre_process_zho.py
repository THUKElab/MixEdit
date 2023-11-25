import re
from typing import List, Union

from bert import tokenization


def remove_space(batch: Union[str, List[str]]):
    def _remove_space(text: str):
        text = text.strip().replace("\u3000", " ").replace("\xa0", " ")
        text = "".join(text.split())
        return text

    if isinstance(batch, str):
        return _remove_space(batch)
    else:
        return [_remove_space(x) for x in batch]


def split_sentence(line: str, flag: str = "all", limit: int = 510):
    """ Split sentences by end dot punctuations
    Args:
        line:
        flag: "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zho":
            # 单字符断句符
            line = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', line)
            # 特殊引号
            line = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', line)
        elif flag == "eng":
            # 英文单字符断句符
            line = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', line)
            # 特殊引号
            line = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', line)
        else:
            # 单字符断句符
            line = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n', line)
            # 特殊引号
            line = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n', line)

        sent_list_ori = line.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except RuntimeError:
        sent_list.clear()
        sent_list.append(line)
    return sent_list


def preprocess(src_list: List[str], file_vocab: str):
    """ Preprocess for Chinese validation sets (MuCGEC, NLPCC)
        1) Split sentence
        2) Tokenization
    """
    # Step 1: Split Sentences
    sub_sents_list = []
    for idx, line in enumerate(src_list):
        line = remove_space(line.rstrip("\n"))
        sents = split_sentence(line, flag="zh")
        if len(line) < 64:
            sub_sents_list.append([line])
        else:
            sub_sents_list.append(sents)

    sent_list, ids = [], []
    for idx, sub_sents in enumerate(sub_sents_list):
        sent_list.extend(sub_sents)
        ids.extend([idx] * len(sub_sents))

    # Step 2: Tokenization
    tok_sent_list = []
    tokenizer = tokenization.FullTokenizer(
        vocab_file=file_vocab,
        do_lower_case=False,
    )

    for sent in sent_list:
        line = remove_space(sent.strip())
        line = tokenization.convert_to_unicode(line)
        if not line:
            return ''
        tokens = tokenizer.tokenize(line)
        tok_sent_list.append(' '.join(tokens))

    return tok_sent_list, ids


