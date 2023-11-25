# File for computing edit Affinity and Diversity
import scipy
from collections import Counter
from data import Dataset


def build_pattern(data: Dataset):
    cnt_edit = Counter()
    for sample in data:
        for src_edits in sample.edits:
            for src_tgt_edits in src_edits:
                for edit in src_tgt_edits:
                    cnt_edit[(" ".join(edit.src_tokens), " ".join(edit.tgt_tokens))] += 1
    return cnt_edit


def calc_edit_affinity(data_ref: Dataset, data_hyp: Dataset):
    """ Compute Edit Distribution Distance between data_ref and data_hyp
        1) Extract two sets of edits
        2) Compute KL Divergence
    """
    patterns_ref = build_pattern(data_ref)
    patterns_hyp = build_pattern(data_hyp)

    dist_ref, dist_hyp = [], []
    for (src_tokens, tgt_tokens), cnt in patterns_ref.items():
        dist_ref.append(cnt)
        dist_hyp.append(patterns_hyp[(src_tokens, tgt_tokens)])
    kl_hyp_ref = scipy.stats.entropy(dist_hyp, dist_ref)
    print(kl_hyp_ref)

    dist_ref, dist_hyp = [], []
    for (src_tokens, tgt_tokens), cnt in patterns_hyp.items():
        dist_hyp.append(cnt)
        dist_ref.append(patterns_ref[(src_tokens, tgt_tokens)])
    kl_ref_hyp = scipy.stats.entropy(dist_ref, dist_hyp)
    print(kl_ref_hyp)

    return 2 / (kl_ref_hyp + kl_hyp_ref)


def calc_edit_diversity(data: Dataset):
    """ Compute Edit Distribution Distance between data_ref and data_hyp
        1) Extract the edit set
        2) Compute joint entropy
    """
    patterns = build_pattern(data)
    dist_edit = list(patterns.values())
    return scipy.stats.entropy(dist_edit)
