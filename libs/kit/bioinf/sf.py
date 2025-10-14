""" This module defines the SequenceFrame class.

It allows to store sequences and their metadata in a pandas DataFrame.
Importantly, it allows different views of the data.

"""


import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from kit.loch.utils import get_seq_hash


class SequenceFrame:
    """Structure to store sequences and their metadata in a pandas DataFrame.

    The following columns are present in df_src and df:
        - pack: the pack of the entry (e.g. train, valid, test, generated, ...)
        - accession: the accession of the entry
        - seq_hash: the sequence hash of the current view
        - seq_src: the original sequence in the source

    The current view stored in df will also have the following columns:
        - seq: the current view of the original sequence. This is a 1:N relationship.
            This means that one sequence in a view can have multiple original sequences
        - c_accession: the number of accessions of the current view
        - c_seq_src: the number of original sequences of the current view
    """

    def __init__(self, translate=("", "", "-"), final_stop_token=False):
        """Initialize the SequenceFrame.

        Prepares the DataFrame to store the original data (self.df_src) and sets a basic view
        on this data where only empyt tokens ('-') are removed from the sequences.

        By this way the sequences "ABC--" and "ABC" would be considered the same in the view.
        The view is stored in self.df.

        Args:
            translate (tuple, optional): The translation table for the sequence.
                Defaults to ("", "", "-").
            final_stop_token (bool, optional): Should a terminal stop token be added.
                Defaults to False.
        """
        self.df_src = pd.DataFrame(columns=["pack", "accession", "seq_hash", "seq_src"])
        self.df = None
        self.set_view(translate, final_stop_token)

    def _convert_seq(self, seq_src):
        trans = str.maketrans(*self.translate)
        seq = ""
        seq_src = seq_src.translate(trans)
        if len(seq_src) > 0:
            if self.final_stop_token:
                seq = seq_src if seq_src[-1] == "*" else (seq_src + "*")
            else:
                seq = seq_src[:-1] if seq_src[-1] == "*" else seq_src
        return seq

    def set_view(self, translate=None, final_stop_token=None):
        """Set the current view of the SequenceFrame.

        The orignal data sequences are aggregated dependent on the applied translation
        (specified with translate) and whether or not a final stop token should be added.

        If two sequences are the same after these transformations, they are considered
        the same in the view and can be found in the same row of self.df
        """
        if translate is not None:
            self.translate = translate
        if final_stop_token is not None:
            self.final_stop_token = final_stop_token

        if len(self.df_src) > 0:
            self.df = self.df_src.copy()
            self.df = self.df.reset_index(drop=False)
            self.df["seq"] = None
            for idx, row in self.df_src.iterrows():
                seq = self._convert_seq(row.seq_src)
                self.df.at[idx, "seq"] = seq
                # check that the seq_hash of the converted sequence
                # is the same as the original seq_hash
                assert self.df.at[idx, "seq_hash"] == get_seq_hash(seq)

            self.df = self.df.groupby(["seq_hash", "seq"]).agg(set).reset_index()
            self.df["c_accession"] = self.df.accession.apply(len)
            self.df["c_seq_src"] = self.df.seq_src.apply(len)

    def __getitem__(self, key):
        return self.df[key]

    def __getattr__(self, attr):
        return getattr(self.df, attr)

    def __len__(self):
        return len(self.df)

    def from_seqs(self, seqs, packs=None, accessions=None, append=False):
        """Fills the SequenceFrame from a list of sequences."""

        df_tmp = pd.DataFrame(
            {
                "pack": packs,
                "accession": accessions,
                "seq_src": seqs,
            }
        )
        df_tmp["seq_hash"] = df_tmp.seq_src.apply(get_seq_hash)
        if accessions is None:
            df_tmp["accession"] = df_tmp.seq_hash

        self.df_src = pd.concat([self.df_src, df_tmp]) if append else df_tmp
        self.set_view()

    def from_fasta(self, fasta_file_path, encoding=None, append=False):
        """Fills the SequenceFrame from a fasta file."""
        self.from_file(fasta_file_path, "fasta", encoding=encoding, append=append)

    def from_phy(self, phy_file_path, encoding=None, append=False):
        """Fills the SequenceFrame from a phylip file."""
        self.from_file(
            phy_file_path, "phylip-relaxed", encoding=encoding, append=append
        )

    def from_file(self, file_path, file_format, encoding=None, append=False):
        """Fills the SequenceFrame from a file."""
        with open(file_path, "r", encoding=encoding) as file:
            seqs, accessions = [], []
            for seq_record in SeqIO.parse(file, file_format):
                seqs.append(str(seq_record.seq))
                accessions.append(seq_record.description)

        self.from_seqs(seqs, accessions=accessions, append=append)

    def to_fasta(self, fasta_file_path, save_src=False):
        """Saves the content of the SequenceFrame to a fasta file.

        Args:
            fasta_file_path (str): The path to the fasta file.
            save_src (bool, optional): Should the original sequences be saved or the view.
                Defaults to False.
        """
        seqs = []
        df = self.df_src if save_src else self.df
        for _, row in df.iterrows():
            seq = row.seq_src if save_src else row.seq
            accession = row.accession if save_src else get_seq_hash(seq)
            seqs.append(SeqRecord(Seq(seq), id=accession, description=""))

        SeqIO.write(seqs, fasta_file_path, format="fasta")
