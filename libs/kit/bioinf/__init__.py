"""This module contains basic constants and functions for working with
protein sequences."""

from collections import defaultdict
import re

import numpy as np
import pandas as pd

# adapted from Bio.PDB.Polypeptide

# special amino acid values
# -... empty position
# X... unknown
# ?... masked language learning

# https://www.ddbj.nig.ac.jp/ddbj/code-e.html

AA1_FULL = "ACDEFGHIKLMNPQRSTVWYU*-X?"
# AA1_FULL = "ACDEFGHIKLMNPQRSTVWYUO*-X?"
AA1_STD = AA1_FULL[:20]
AA1_ALIASES = {"B": "N", "Z": "Q"}
NEXT_CHAIN = '/'

unpredicted_tokens = "UX?"  # the tokens that our models should not predict

AA_CHARACTERISTICS_1 = {
    'charged': "RKHDE",
    'polar': "STNQ",
    'hydrophobic': "AVILMFYW",
    'special': "CGP"
}

AA_CHARACTERISTICS_2 = {
    'aliphatic': "AGILMPV",  # non-polar
    'aromatic': "FWY",  # non-polar
    'polar': "NCQST", # polar
    'acidic': "DE",  # polar
    'basic': "RHK",  # polar
}

AA_CHARACTERISTICS_2a = {
    'non-polar': AA_CHARACTERISTICS_2['aliphatic'] + AA_CHARACTERISTICS_2['aromatic'],
    'polar': AA_CHARACTERISTICS_2['polar'] + AA_CHARACTERISTICS_2['acidic'] + AA_CHARACTERISTICS_2['basic']
}


# additional proteinogenic AAs
# https://www.ebi.ac.uk/chebi/searchId.do?printerFriendlyView=true&chebiId=83813&structureView=
AA3_FULL = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",  # standard AAs
    "SEC",  # "PYL" # additional proteinogenic AAs
    "STP",  # stop token
    "---",
    "XXX",
    "???",
]  # special values (None, Unknown, Masked)
AA3_STD = AA3_FULL[:20]

N_AA_FULL = len(AA1_FULL)
N_AA_STD = len(AA1_STD)

AA1_TO_IDX = {}
IDX_TO_AA1 = {}
AA3_TO_IDX = {}
IDX_TO_AA3 = {}
AA3_TO_AA1 = {}
AA1_TO_AA3 = {}

# Create some lookup tables
for i in range(0, N_AA_FULL):
    aa_1 = AA1_FULL[i]
    aa_3 = AA3_FULL[i]

    AA1_TO_IDX[aa_1] = i
    IDX_TO_AA1[i] = aa_1
    AA3_TO_IDX[aa_3] = i
    IDX_TO_AA3[i] = aa_3
    AA3_TO_AA1[aa_3] = aa_1
    AA1_TO_AA3[aa_1] = aa_3


def only_std_aa(seqs, ignore="*-"):
    """Checks whether sequences only contain standard amino acids.

    :param seqs: all sequences to consider (list or a single string)
    :param ignore: str - all amino acids to ignore
    """

    if isinstance(seqs, list):
        result = []
        for seq in seqs:
            result.append(only_std_aa(seq, ignore))
        return result

    check = AA1_STD + list(ignore)
    for letter in seqs:
        if letter not in check:
            return False
    return True


def get_kmers(seqs, lengths, check_aa=True):
    """returns a set with all kmers of given lengths in the sequnces

    :param seqs: all sequences to consider (list or a single string)
    :param lengths: all kmer lengths to consider (list or single number)
    :param check_aa: if True, only kmers with standard amino acids are returned
    :return: set of kmers in the sequences
    """

    result = set()
    if isinstance(seqs, list):
        for seq in seqs:
            result |= get_kmers(seq, lengths, check_aa)
    else:
        seq = seqs
        if isinstance(lengths, list):
            for length in lengths:
                result |= get_kmers(seq, length, check_aa)
        else:
            length = lengths
            result = {str(seq[x : x + length]) for x in range(len(seq) - length + 1)}
            if check_aa:
                result = {x for x in result if all(y in AA1_STD for y in x)}

    return result

def get_kmers_list(seq, lengths):
    """returns a list with all kmers of given lengths in the sequnce

    :param seq: str - sequence
    :param lengths: list - all kmer lengths to consider
    :return: list of kmers in the sequence
    """

    result = []
    for length in lengths:
        result.extend([seq[x : x + length] for x in range(len(seq) - length + 1)])
    return result


def seq_to_kmers_list(seq, k):
    """Returns a list of all kmers of length k in the sequence.
    In contrast to get_kmers, this function does returns kmers that occure
    several times multiple times."""

    kmers = []
    for pos in range(len(seq) - k + 1):
        kmers.append(seq[pos : pos + k])
    return kmers


def generate_random_aa_seq(length):
    """Generates a random amino acid sequence of given length.
    The standard amino acids are drawn with equal probability."""

    return "".join(np.random.choice(list(AA1_STD), length))


def seq_to_idx(seq):
    """Converts a sequence of amino acids to a list of indices."""

    seq = [
        (aa if aa not in AA1_ALIASES else AA1_ALIASES[aa]) for aa in seq
    ]  # replace aliases
    idx = [AA1_TO_IDX[aa] for aa in seq]
    return idx


def idx_to_seq(idx):
    """Converts a list of indices to a sequence of amino acids."""

    return "".join([IDX_TO_AA1[i.item()] for i in idx])


def tensor_to_seqs(tensor):
    """Converts a tensor of indices to a list of sequences."""

    result = []
    for t in tensor:
        result.append(idx_to_seq(t))
    return result


def seq_dict_to_df(d_sequences):
    """Converts a dictionary with sequences as keys and accessions as values
    to a dataframe with the sequences as indices and the accessions as a list.
    Also adds columns for the sequence length, the number of accessions and
    a boolean column indicating whether the sequence is complete (no 'X') and
    a boolean column indicating whether the sequence only contains standard
    amino acids.

    :param d_sequences: dict - dictionary with sequences as keys and accessions as values
    :return: pd.DataFrame - dataframe"""

    seqs = []
    accessions = []
    cnts = []
    lengths = []
    complete = []
    stdaa = []
    for key, value in d_sequences.items():
        seqs.append(key)
        accessions.append(value)
        cnts.append(len(value))
        lengths.append(len(key))
        complete.append("X" not in key)
        stdaa.append(all(r in (AA1_STD + "*") for r in key))

    return pd.DataFrame(
        {
            "seq": seqs,
            "length": lengths,
            "accession": accessions,
            "cnt": cnts,
            "complete": complete,
            "stdaa": stdaa,
        }
    ).set_index("seq")
