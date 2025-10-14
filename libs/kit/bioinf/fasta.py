"""Functions for reading and writing fasta files."""

from collections import defaultdict
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from kit.loch.utils import get_seq_hash
from kit.bioinf import seq_dict_to_df
from kit.path import join


def fasta_to_df(filename, encoding=None):
    seqs, descriptions = [], []
    with open(filename, "r", encoding=encoding) as file:
        for seq_record in SeqIO.parse(file, "fasta"):
            seqs.append(str(seq_record.seq))
            descriptions.append(seq_record.description)
    return pd.DataFrame({'seq': seqs, 'description': descriptions})


def read_fasta(
    filenames, encoding=None, stop_token=True, return_df=False, evaluate=False
):
    """Reads a list of fasta files.

    Args:
        dir_path (str): Path to the folder.
        encoding (str, optional): Encoding of the fasta files. Defaults to None.
        stop_token (bool, optional): Should a stop token be added to the sequences.
            Defaults to True.
        return_df (bool, optional): Should the sequences be returned as a dataframe.
            Or as a dictionary. Defaults to False (a dictionary is returned).
            The dictionary has the sequences as keys and a list of accessions as values.
            The dataframe has the sequences as indices (see kit.bioinf.seq_dict_to_df)
        evaluate (bool, optional): Should the description be evaluated before using is
            as accession. Defaults to False.
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    sequences = defaultdict(lambda: [])
    for filename in filenames:
        with open(filename, "r", encoding=encoding) as file:
            for seq_record in SeqIO.parse(file, "fasta"):
                seq = str(seq_record.seq)
                if len(seq) > 0:
                    if stop_token and seq[-1] != "*":
                        seq += "*"
                    if not stop_token and seq[-1] == "*":
                        seq = seq[:-1]
                description = (
                    eval(seq_record.description) if evaluate else seq_record.description
                )
                if isinstance(description, list):
                    sequences[seq] += description
                else:
                    sequences[seq].append(str(description))

    d_sequences = dict(sequences)

    if return_df:
        return seq_dict_to_df(d_sequences)
    return d_sequences


def write_fasta(filename, df_sequences, translate=("", "", "")):
    """Write a dataframe with sequences to a fasta file.

    :param filename: str - path to output file
    :param df_sequences: pd.DataFrame - dataframe with sequences
    :param translate: tuple - (src, tgt, None) for translation of sequences"""
    seqs = []
    for seq, row in df_sequences.iterrows():
        new_seq = seq.translate(str.maketrans(*translate))
        new_id = str(row.accession)
        seqs.append(SeqRecord(Seq(new_seq), id=new_id, description=""))

    SeqIO.write(seqs, filename, format="fasta")


def seqs_to_fasta(seqs, filename, accessions=None, translate=("", "", "")):
    """Write a list of sequences to a fasta file.

    :param filename: str - path to output file
    :param seqs: list/str - list of sequences or a single sequence string
    :param translate: tuple - (src, tgt, None) for translation of sequences
    """

    if isinstance(seqs, str):
        seqs = [seqs]
        if accessions is not None:
            accessions = [accessions]

    if accessions is None:
        accessions = [get_seq_hash(seq) for seq in seqs]

    df = pd.DataFrame({"seq": seqs, "accession": accessions}).set_index("seq")
    write_fasta(filename, df, translate=translate)


def fastas_to_seqs(fasta_file_paths, stop_token=True):
    """Reads a list of fasta files into a list of sequences

    Args: see read_fasta
    """

    d_seqs = read_fasta(fasta_file_paths, return_df=False, evaluate=False, stop_token=stop_token)
    return list(d_seqs.keys())


def translate_fasta(input_file_path, output_file_path=None, translate=("", "", "")):
    """Applies a string translation to a fasta file."""

    seq = fastas_to_seqs(input_file_path)
    seqs_to_fasta(
        seq,
        output_file_path if output_file_path is not None else input_file_path,
        translate=translate,
    )
