import copy
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union, Optional

from utils import get_logger, remove_space

LOGGER = get_logger(__name__)
DELIMITER_M2 = "|||"
EDIT_NONE_TYPE = {"noop", "NA"}
EDIT_NONE_CORRECTION = {"-NONE-"}


@dataclass
class Edit(object):
    tgt_idx: int = field(
        default=0, metadata={"help": "Target index"}
    )
    src_interval: Optional[List[int]] = field(
        default=None, metadata={"help": "Source interval"}
    )
    tgt_interval: Optional[List[int]] = field(
        default=None, metadata={"help": "Target interval"}
    )
    src_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "Source tokens"}
    )
    tgt_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "Target tokens"}
    )
    src_tokens_tok: Optional[Any] = field(
        default=None, metadata={"help": "Source tokens tokenized by third toolkit"}
    )
    tgt_tokens_tok: Optional[Any] = field(
        default=None, metadata={"help": "Target tokens tokenized by third toolkit"}
    )
    type: Optional[List[str]] = field(
        default=None, metadata={"help": "Edit type"}
    )


@dataclass
class Sample(object):
    index: int = field(
        default=None, metadata={"help": "Sample Index"}
    )
    source: List[str] = field(
        default=None, metadata={"help": "Source sentences, which are usually ungrammatical"}
    )
    target: List[str] = field(
        default=None, metadata={"help": "Target sentences, which are grammatical"}
    )
    _edits: Optional[List[List[List[Edit]]]] = field(
        default=None, metadata={"help": "Edits"}
    )

    @property
    def edits(self):
        return self._edits

    def contains_empty(self):
        return any([not x for x in self.source]) or any([not x for x in self.target])

    def __repr__(self):
        return f"Sample(index={self.index}, source={self.source}, target={self.target})"

    def __deepcopy__(self, memodict={}):
        return Sample(
            index=self.index,
            source=self.source.copy(),
            target=self.target.copy(),
            _edits=copy.deepcopy(self._edits),
        )


@dataclass
class Dataset(object):
    samples: List[Sample] = field(default_factory=list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def __iter__(self):
        return iter(self.samples)

    def append(self, sample):
        self.samples.append(sample)

    def extend(self, dataset):
        orig_len = len(self)
        self.samples.extend(dataset.samples)
        for sample_idx in range(orig_len, len(self.samples)):
            self.samples[sample_idx].index = sample_idx

    def merge(self, dataset):
        assert len(self) == len(dataset)
        for sample1, sample2 in zip(self, dataset):
            assert len(sample1.source) == len(sample2.source) == 1
            assert sample1.source[0] == sample2.source[0]
            sample1.target.extend(sample2.target)
            sample1.edits[0].extend(sample2.edits[0])


@dataclass
class DataReader(ABC):
    remove_space: bool = field(
        default=False, metadata={"help": "Remove spaces in the sentence"}
    )
    sort_by_len: bool = field(
        default=False, metadata={"help": "Sort samples by length"}
    )

    @abstractmethod
    def read(
            self, file_input: Union[str, List[str]],
            max_sample: int = -1,
            max_target: int = -1,
    ) -> Dataset:
        raise NotImplementedError()

    def read_post(self, samples: List[Sample], filename: str) -> Dataset:
        filename = os.path.basename(filename)
        LOGGER.info(f"{self.__class__}: Read {len(samples)} samples from {filename}.")
        if self.remove_space:
            for sample in samples:
                sample.source = remove_space(sample.source)
                sample.target = remove_space(sample.target)
        if self.sort_by_len:
            samples = sorted(samples, key=lambda x: len(x.source[0]))
        return Dataset(samples=samples)


class M2DataReader(DataReader):
    def read(
            self, file_input: str,
            max_sample: int = -1,
            max_target: int = -1,
    ) -> Dataset:
        data, tgt_tokens_list, edit_lines_list, edit_objs_list = [], [], [], []
        curr_src_tokens = None
        for src_tokens, tgt_tokens, edit_lines, edit_objs, tgt_idx in self.read_m2_file(
                file_input,
                max_sample=max_sample,
                max_target=max_target,
        ):
            if curr_src_tokens is None:
                curr_src_tokens = src_tokens

            if tgt_idx == len(tgt_tokens_list):  # Same sample
                tgt_tokens_list.append(tgt_tokens)
                edit_lines_list.append(edit_lines)
                edit_objs_list.append(edit_objs)
            else:  # Next sample
                data.append(Sample(
                    index=len(data),
                    source=[" ".join(curr_src_tokens)],
                    target=[" ".join(x) for x in tgt_tokens_list],
                    _edits=[edit_objs_list.copy()],
                ))
                tgt_tokens_list, edit_lines_list, edit_objs_list = [], [], []
                curr_src_tokens = src_tokens
                tgt_tokens_list.append(tgt_tokens)
                edit_lines_list.append(edit_lines)
                edit_objs_list.append(edit_objs)

        if tgt_tokens_list:
            data.append(Sample(
                index=len(data),
                source=[" ".join(curr_src_tokens)],
                target=[" ".join(x) for x in tgt_tokens_list],
                _edits=[edit_objs_list.copy()],
            ))
        return self.read_post(data, file_input)

    def read_m2_file(
            self,
            m2_file: str,
            max_sample: int = -1,
            max_target: int = -1,
    ):
        num_target, num_sample, line_idx = 0, 0, 0
        src_sent, src_tokens, edit_lines = "", [], []
        with open(m2_file, "r", encoding="utf8") as f:
            m2_lines = f.readlines()

        while line_idx < len(m2_lines):
            if 0 <= max_sample <= num_sample:  break
            line = m2_lines[line_idx].strip()

            if line.startswith("S"):  # Source line
                if line.startswith("S "):
                    src_sent = line.replace("S ", "", 1)
                    src_tokens = src_sent.split()
                else:
                    src_sent = ""
                    src_tokens = []
                line_idx += 1

            elif line.startswith("T"):  # Target line
                if line.endswith("没有错误") or line.endswith("无法标注"):
                    line_idx += 1
                    LOGGER.debug(f"Unchanged sentence: {src_sent}")
                if int(line.split("-", 1)[1][1]) != 0:
                    # Only happen on ChERRANT (Chinese). We ignore the follow-up edits.
                    LOGGER.info(f"Ignore repetitive target: {line}")
                    while m2_lines[line_idx].startswith("A "):
                        line_idx += 1
                    continue

            elif line.startswith("A"):  # Editorial line
                line = line.replace("A ", "", 1)
                tgt_idx = int(line.rsplit(DELIMITER_M2, 1)[-1])
                if tgt_idx != num_target:  # New Target
                    assert tgt_idx == num_target + 1, f"Error Parsing: Source={src_sent}, tgt_idx={tgt_idx}"
                    if max_target <= 0 or num_target < max_target:
                        tgt_tokens, edit_objs = self.build_target(src_tokens, edit_lines)
                        yield src_tokens, tgt_tokens, edit_lines.copy(), edit_objs, num_target
                    num_target += 1
                    edit_lines.clear()
                line_idx += 1
                edit_lines.append(line)

            elif not line:  # New target
                if max_target <= 0 or num_target < max_target:
                    tgt_tokens, edit_objs = self.build_target(src_tokens, edit_lines)
                    yield src_tokens, tgt_tokens, edit_lines.copy(), edit_objs, num_target
                while line_idx < len(m2_lines) and not m2_lines[line_idx].strip():
                    line_idx += 1
                if line_idx == len(m2_lines):
                    break
                num_sample += 1
                num_target = 0
                edit_lines.clear()

            if line and line_idx == len(m2_lines) and max_target < 0 or num_target < max_target:
                tgt_tokens, edit_objs = self.build_target(src_tokens, edit_lines)
                yield src_tokens, tgt_tokens, edit_lines.copy(), edit_objs, num_target

    @classmethod
    def build_target(cls, src_tokens: List[str], m2_lines: List[str] = None) -> Tuple[List[str], List[Edit]]:
        edits = []
        src_offset, src_tokens = 0, src_tokens.copy()
        tgt_offset, tgt_tokens = 0, src_tokens.copy()
        for m2_line in m2_lines:
            if m2_line.startswith("A "):
                m2_line = m2_line.replace("A ", "", 1)
            elements = m2_line.split(DELIMITER_M2, 2)
            elements = elements[:2] + elements[-1].rsplit(DELIMITER_M2, 3)
            assert len(elements) == 6, f"Error Parsing: {m2_line}"

            src_beg_idx, src_end_idx = map(int, elements[0].split())
            # Ignore certain edits
            if elements[1] in EDIT_NONE_TYPE:
                assert src_beg_idx == src_end_idx == -1 and elements[2] in EDIT_NONE_CORRECTION
                continue

            edit_src_tokens = src_tokens[src_beg_idx:src_end_idx]
            edit_tgt_tokens = elements[2].strip().split() if elements[2] not in EDIT_NONE_CORRECTION else []

            tgt_beg_idx = src_beg_idx + tgt_offset
            tgt_end_idx = tgt_beg_idx + len(edit_tgt_tokens)
            tgt_tokens[tgt_beg_idx: src_end_idx + tgt_offset] = edit_tgt_tokens
            tgt_offset += len(edit_tgt_tokens) - len(edit_src_tokens)

            edits.append(Edit(
                int(elements[5]),
                src_interval=[src_beg_idx, src_end_idx],
                tgt_interval=[tgt_beg_idx, tgt_end_idx],
                src_tokens=edit_src_tokens.copy(),
                tgt_tokens=edit_tgt_tokens.copy(),
                type=[elements[1]],
            ))
            LOGGER.debug(f"Build Edit: {edits[-1]}")
            # Sanity Check
            assert (
                    tgt_beg_idx == tgt_end_idx or
                    tgt_tokens[tgt_beg_idx: tgt_end_idx] == edit_tgt_tokens
            ), f"Error Parsing: {' '.join(src_tokens)} || {' '.join(tgt_tokens)}"
        return tgt_tokens, edits


def apply_edits(src_tokens: List[str], edits: List[Edit], strict=True) -> List[str]:
    """ Apply Edits on src_tokens """
    tgt_offset, tgt_tokens = 0, src_tokens.copy()
    for edit in edits:
        src_beg_idx, src_end_idx = edit.src_interval[0], edit.src_interval[1]

        if strict and edit.src_tokens != src_tokens[src_beg_idx: src_end_idx]:
            raise ValueError(f"Inconsistent Edit: {edit} || {src_tokens[src_beg_idx: src_end_idx]}")
        elif edit.src_tokens != src_tokens[src_beg_idx: src_end_idx]:
            LOGGER.warning(f"Warning - Inconsistent Edit: {edit} || {src_tokens[src_beg_idx: src_end_idx]}")

        tgt_beg_idx = src_beg_idx + tgt_offset
        tgt_end_idx = tgt_beg_idx + len(edit.tgt_tokens)
        tgt_tokens[tgt_beg_idx: src_end_idx + tgt_offset] = edit.tgt_tokens
        tgt_offset += len(edit.tgt_tokens) - len(edit.src_tokens)

        # Sanity Check
        if tgt_tokens[tgt_beg_idx: tgt_end_idx] != edit.tgt_tokens:
            raise ValueError(f"{' '.join(src_tokens)} || {' '.join(tgt_tokens)}")
    return tgt_tokens
