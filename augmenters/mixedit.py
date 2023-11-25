import math
import json
import random
import logging
from collections import Counter, defaultdict
from typing import Callable

from .data import Edit, Sample, Dataset, M2DataReader, apply_edits
from utils import get_logger

LOGGER = get_logger(__name__, level=logging.INFO)


class MixEditAugmenter:
    """ This Augmenter is introduced in the paper:
        MixEdit: Revisiting Data Augmentation and Beyond for Grammatical Error Correction [EMNLP 2023]
        MixEdit aims to strategically and dynamically augments realistic data.
    """

    def __init__(
            self,
            temperature: float = 1.0,
            enable_filter_pattern: bool = True,
            filter_pattern_min_freq: int = 3,
            filter_pattern_max_diff: int = 2,
            min_pattern: int = 2,
            pattern_noise_rate: float = 0.0,
            pattern_noise_step: int = 10000,
            file_pattern: str = None,
            verbose: bool = True,
            encode_fn: Callable = None,
            decode_fn: Callable = None,
            remove_bpe: str = None,
    ):
        super().__init__()
        self.reader = M2DataReader()
        self.temperature = temperature
        self.enable_filter_pattern = enable_filter_pattern
        self.filter_pattern_min_freq = filter_pattern_min_freq
        self.filter_pattern_max_diff = filter_pattern_max_diff
        self.min_pattern = min_pattern
        self.pattern_noise_rate = pattern_noise_rate
        self.pattern_noise_step = pattern_noise_step
        self.file_pattern = file_pattern
        self.verbose = verbose
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.remove_bpe = remove_bpe

        self.num_updates = 0
        self.patterns = None
        self.patterns_prob = None
        self.tgt2sample = None

    def setup(
            self, data: Dataset = None,
            write_pattern: str = None,
            build_tgt2sample: bool = False,
    ):
        """ Extract Edits from data
            1) Build edits
            2) Extract edit patterns
        """
        if self.patterns_prob is not None:
            return data

        # Read or build pattern
        if self.file_pattern:
            self.patterns = self.read_file_pattern(self.file_pattern)
        else:
            assert data is not None
            self.patterns = self.build_pattern(data)

        if write_pattern is not None:
            with open(write_pattern, "w", encoding="utf-8") as f:
                json.dump(self.patterns, f)

        # Filter pattern
        if self.enable_filter_pattern:
            self.patterns = self.filter_pattern(
                self.patterns,
                filter_pattern_min_freq=self.filter_pattern_min_freq,
                filter_pattern_max_diff=self.filter_pattern_max_diff,
            )

        # Build pattern Distribution
        self.patterns_prob = self.build_pattern_distribution(
            patterns=self.patterns,
            temperature=self.temperature,
        )

        if self.verbose:
            self.print_pattern_info()

        if build_tgt2sample:
            self.tgt2sample = defaultdict(list)
            for sample in data:
                assert len(sample.target) == 1
                self.tgt2sample[sample.target[0]].append(sample)
        return data

    @classmethod
    def read_file_pattern(cls, file_pattern: str):
        LOGGER.info(f"Read pattern from: {file_pattern}")
        with open(file_pattern, "r", encoding="utf-8") as f:
            patterns = json.load(f)
        return patterns

    @classmethod
    def build_pattern(cls, data: Dataset):
        LOGGER.info(f"Build pattern from dataset with {len(data)} samples")
        cnt_edit = Counter()
        for sample in data:
            for src_edits in sample.edits:
                for src_tgt_edits in src_edits:
                    for edit in src_tgt_edits:
                        cnt_edit[(" ".join(edit.src_tokens), " ".join(edit.tgt_tokens))] += 1

        patterns = defaultdict(list)
        for (src_tokens, tgt_tokens), cnt in cnt_edit.items():
            patterns[tgt_tokens].append((src_tokens, cnt))
        return patterns

    @classmethod
    def filter_pattern(
            cls,
            patterns,
            filter_pattern_min_freq,
            filter_pattern_max_diff,
    ):
        """ Filter pattern if one of the followings is satisfied:
            1) The length of source and target differ too much
            2) The frequency of pattern is low
        """
        num_filter = 0
        new_patterns = defaultdict(list)

        for tgt, pat in patterns.items():
            for src, cnt in pat:
                src_len = len(src.split())
                tgt_len = len(tgt.split())
                is_filter = False

                # if filter_pattern_min_freq > 0 and filter_pattern_max_diff > 0:
                #     if cnt < filter_pattern_min_freq or abs(src_len - tgt_len) > filter_pattern_max_diff:
                #         is_filter = True
                # else:

                if abs(src_len - tgt_len) > 4:
                    is_filter = True
                elif abs(src_len - tgt_len) > 2 and cnt < 3:
                    is_filter = True

                if is_filter:
                    num_filter += 1
                else:
                    new_patterns[tgt].append((src, cnt))

        LOGGER.info(
            f"Filter pattern: {num_filter}, from "
            f"{sum([len(pat) for tgt, pat in patterns.items()])} to "
            f"{sum([len(pat) for tgt, pat in new_patterns.items()])}"
        )
        return new_patterns

    @classmethod
    def build_pattern_distribution(cls, patterns, temperature: float = 1.0):
        """ Construct pattern distribution
            1) Calculate likelihood p(x_i | y_i), where x_i is an error, y_i is target
            2) Reshape pattern distribution
        """
        # Calculate likelihood p(x_i | y_i)
        patterns_prob = defaultdict(list)
        for tgt_tokens, corruption in patterns.items():
            sum_freq = sum([x[1] for x in corruption])
            for src_tokens, cnt in corruption:
                patterns_prob[tgt_tokens].append((src_tokens, cnt / sum_freq))

            # Reshape Pattern Distribution
            if abs(temperature - 1.0) > 1e-9:
                sum_prob = sum([math.pow(x[1], temperature) for x in patterns_prob[tgt_tokens]])
                for idx, (src_tokens, prob) in enumerate(patterns_prob[tgt_tokens]):
                    patterns_prob[tgt_tokens][idx] = (
                        src_tokens,
                        math.pow(prob, temperature) / sum_prob,
                    )
                    LOGGER.debug(
                        f"Reshape prob: ({tgt_tokens}, {src_tokens}) "
                        f"{round(prob, 4)} -> {round(patterns_prob[tgt_tokens][idx][1], 4)}",
                    )
        return patterns_prob

    def print_pattern_info(self):
        num_edit, num_pattern = 0, 0
        cnt_pattern = Counter()
        for tgt_tokens, patterns in self.patterns.items():
            sum_freq = sum([x[1] for x in patterns])
            num_edit += sum_freq
            num_pattern += len(patterns)
            cnt_pattern[len(patterns)] += 1
        LOGGER.info(f"Total {num_edit} edits, {num_pattern} patterns")
        LOGGER.info(f"Pattern {cnt_pattern}")

    def pattern_noise(
            self, sample: Sample,
            pattern_noise_rate: float,
            max_span_len: int = 3,
            corrupt_target: bool = False,
    ) -> Sample:
        """ Randomly corrupt tokens according to error patterns """
        new_sample = Sample(
            source=sample.target.copy() if corrupt_target else sample.source.copy(),
            target=sample.target.copy(),
        )
        for src_idx, src_str in enumerate(new_sample.source):
            src_tokens = src_str.strip().split()
            new_src_tokens = src_tokens.copy()
            idx, offset = 0, 0
            while idx < len(src_tokens):
                if random.random() < pattern_noise_rate:
                    span_len = max_span_len
                    span = " ".join(src_tokens[idx: idx + span_len])
                    while not span in self.patterns_prob and span_len > 0:
                        span_len -= 1
                        span = " ".join(src_tokens[idx: idx + span_len])

                    cand_span = [x[0] for x in self.patterns_prob[span]]
                    cand_prob = [x[1] for x in self.patterns_prob[span]]
                    hit_span = random.choices(cand_span, cand_prob)[0].split()

                    new_src_tokens[idx + offset: idx + offset + span_len] = hit_span
                    idx += max(1, len(span.split()))
                    offset += len(hit_span) - span_len
                idx += 1
            new_sample.source[src_idx] = " ".join(new_src_tokens)
            LOGGER.debug(f"Pattern Noise: {src_str} || {new_sample.source[src_idx]}")
        return new_sample

    def augment_sample(self, sample: Sample, num_augmentation: int = 1):
        assert len(sample.source) == 1 and len(sample.target) == 1
        src, tgt, edits = sample.source[0], sample.target[0], sample.edits[0][0]

        # Draw alternative edits from the Error Pattern Pool
        new_src_list = []
        for _ in range(num_augmentation):
            edits_tgt2src = []
            for e in edits:
                src_tokens, tgt_tokens = e.src_tokens, e.tgt_tokens
                src_tokens_str, tgt_tokens_str = " ".join(src_tokens), " ".join(tgt_tokens)

                if tgt_tokens_str not in self.patterns_prob:
                    LOGGER.debug(f"Not exist in pattern: {tgt_tokens_str}")
                    continue

                cand_src = [x[0] for x in self.patterns_prob[tgt_tokens_str]]
                cand_prob = [x[1] for x in self.patterns_prob[tgt_tokens_str]]
                hit_src = random.choices(cand_src, cand_prob)[0]

                # Construct new edit, note that src and tgt are reverse
                edit_tgt2src = Edit(
                    tgt_idx=0,
                    src_interval=e.tgt_interval,
                    tgt_interval=None,
                    src_tokens=tgt_tokens,
                    tgt_tokens=hit_src.split(),
                )
                edits_tgt2src.append(edit_tgt2src)

                LOGGER.debug(f"Replace `{src_tokens_str}` with `{hit_src}`")
                LOGGER.debug(edit_tgt2src)

            # Reconstruct source
            new_src_tokens = apply_edits(tgt.split(), edits_tgt2src)
            new_src_str = " ".join(new_src_tokens)
            new_src_list.append(new_src_str)
            LOGGER.debug(f"Target    : {tgt}")
            LOGGER.debug(f"Source    : {src}")
            LOGGER.debug(f"New Source: {new_src_str}")

        result = Sample(source=new_src_list, target=[tgt])

        # Pattern Noise
        if self.pattern_noise_step <= 0:
            pattern_noise_rate = self.pattern_noise_rate
        else:
            pattern_noise_rate = max(
                0.0, self.pattern_noise_rate -
                     self.num_updates * (self.pattern_noise_rate / (self.pattern_noise_step + 1e-9)),
            )
        if abs(pattern_noise_rate) > 1e-9:
            result = self.pattern_noise(result, pattern_noise_rate)
        return result

    def augment_sample_for_fairseq(
            self,
            tgt_tokens,
            src_dict,
            tgt_dict=None,
    ):
        """ MixEdit
            1) Decode to string
            2) Invoke MixEditAugmenter.augment_sample
            3) Encode to tensor
        """
        assert self.tgt2sample is not None, f"Assure `build_tgt2sample = True` for setUp()"
        tgt_dict = src_dict if tgt_dict is None else tgt_dict

        def find_sample(target: str):
            if target not in self.tgt2sample:
                if "<unk>" not in target:
                    LOGGER.warning(f"Not exist in tgt2sample: {target}")
                return Sample(source=[target], target=[target])
            cands = self.tgt2sample[target]
            return cands[random.randint(0, len(cands) - 1)]

        # Decode to string
        tgt_tensor = tgt_tokens
        tgt_bpe = tgt_dict.string(tgt_tensor)
        tgt_str = self.decode_fn(tgt_bpe)
        if self.remove_bpe:
            tgt_str = tgt_str.replace(self.remove_bpe, "")

        # Find Sample object from self.training_dataset
        sample = find_sample(tgt_str)

        # Invoke MixEditAugmenter
        new_src = self.augment_sample(sample).source[0]

        # Encode to tensor
        src_tensor = src_dict.encode_line(
            self.encode_fn(new_src),
            append_eos=True,
            add_if_not_exist=False,
        ).type_as(tgt_tensor)
        return src_tensor
