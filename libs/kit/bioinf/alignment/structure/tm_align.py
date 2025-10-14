""" This module contains functions for aligning two structures 
using the TMalign algorithm. """

import os
import re
import subprocess


def align_structures(ref_pdb_file_path, pdb_file_path, output=None, return_chain_lengths=False):
    """aligns two structures using the TMalign algorithm.

    :param pdb: str - path to the structure to be aligned
    :param pdb_ref: str - path to the reference structure
    :param output: str - path to the output file (Optional)
    :return: tuple - (TM-score, aligned_length, RMSD, identical)
    """

    if not os.path.exists(ref_pdb_file_path):
        raise Exception(f"TMalign input file {ref_pdb_file_path} does not exist")
    if not os.path.exists(pdb_file_path):
        raise Exception(f"TMalign input file {pdb_file_path} does not exist")

    if output is not None:
        command = ["TMalign", pdb_file_path, ref_pdb_file_path, "-o", output]
    else:
        command = ["TMalign", pdb_file_path, ref_pdb_file_path]

    result = subprocess.run(command, capture_output=True, check=False)
    res = result.stdout.decode("utf-8")

    tm_score, aligned_length, rmsd, identical = None, None, None, None
    len_chain1, len_chain2 = None, None
    for line in res.split("\n"):
        # read TM score
        regres = re.findall(
            r"^TM-score=\s+(\d\.\d+)\s\(if normalized by length of Chain_2,", line
        )
        if len(regres) == 1 and tm_score is None:
            tm_score = float(regres[0])

        # read aligned length, RMSD and identical
        regres = re.findall(
            r"Aligned length=\s+(\d+), RMSD=\s+(\d+\.\d+), "
            + r"Seq_ID=n_identical/n_aligned=\s+(\d+\.\d+)",
            line,
        )
        if len(regres) == 1:
            aligned_length = int(regres[0][0])
            rmsd = float(regres[0][1])
            identical = float(regres[0][2])

        # read chain lengths
        regres = re.findall(r"^Length of Chain_1:\s+(\d+) residues", line)
        if len(regres) == 1:
            len_chain_1 = float(regres[0])
        regres = re.findall(r"^Length of Chain_2:\s+(\d+) residues", line)
        if len(regres) == 1:
            len_chain_2 = float(regres[0])

    if return_chain_lengths:
        result = tm_score, aligned_length, rmsd, identical, len_chain_1, len_chain_2
    else:
        result = tm_score, aligned_length, rmsd, identical

    return result
