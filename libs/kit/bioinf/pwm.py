import os
import copy
import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kit.path import join
from kit.bioinf import AA1_STD, N_AA_STD, AA1_TO_IDX



def count_amino_acid_occurrences(peptides):
    """ Count how often each AA is observed at each position

    param peptide: list - list of peptides
    return: dict - dictionary of numpy arrays (key: length of peptides, value: numpy array of counts)
    """

    counts = {}

    for peptide in tqdm(peptides):
        length = len(peptide)
        if length not in counts:
            counts[length] = np.zeros((N_AA_STD, length))

        for pos, aa in enumerate(peptide):
            counts[length][AA1_STD.find(aa), pos] += 1

    for length, np_array in counts.items():
        counts[length] = pd.DataFrame(
            columns=list(range(length)),
            index=list(AA1_STD),
            data=np_array,
        )
        counts[length].index.name = "AA"    
    return counts


def calc_PWMs(counts):
    PWMs = {}
    PWMs_log = {}
    for length, _pwm in counts.items():
        PWMs[length] = _pwm.div(_pwm.sum(axis=0), axis=1)
        if not np.allclose(PWMs[length].sum(axis=0), 1.):
            raise ValueError(f"Sum of probabilities for length {length} is not 1")
        
        np.seterr(divide="ignore")
        PWMs_log[length] = np.log(PWMs[length])
        np.seterr(divide="warn")
        PWMs_log[length][PWMs_log[length] == -np.inf] = -1e9
        
    return PWMs, PWMs_log


def save_PWMs(allele_or_name, PWMs, PWMs_log, folder):
    lengths = set(PWMs.keys())
    if len(lengths ^ set(PWMs_log.keys())) > 0:
        raise ValueError("Lengths for PWMs and PWMs_log are not the same")

    for_path = allele_or_name.replace("*", "_")

    for length in tqdm(lengths, "generate PWMs for each length", leave=False):
        # save the produced PWMs
        PWMs[length].to_csv(
            join(
                folder,
                "pwm",
                for_path,
                f"{for_path}-{length}.csv",
            )
        )
        PWMs_log[length].to_csv(
            join(
                folder,
                "pwm",
                for_path,
                f"{for_path}-{length}_log.csv",
            )
        )
