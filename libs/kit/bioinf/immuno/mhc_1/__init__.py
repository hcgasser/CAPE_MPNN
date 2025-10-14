"""This module specifies the base class for MHC Class 1 binding predictors"""

import os
import re
from collections import defaultdict
import importlib

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from kit.path import join
from kit.loch.utils import get_seq_hash
from kit.bioinf import get_kmers, seq_to_kmers_list
from kit.log import log_info, log_warning

from kit.bioinf.immuno.utils import get_mhc_name_split


MHC_1_PEPTIDE_LENGTHS = [8, 9, 10]
MHC_2_pep_lengths = [13, 14, 15, 16, 17]


class Mhc1Predictor:
    """Base class for MHC Class 1 binding predictors

    :param name: str - name of the predictor
    :param data_dir_path: str - path to the directory with the predictor data
        this can include already predicted peptides and percentiles
    :param limit: float - percentile limit for the predictor (e.g. 0.02)
        values below this limit are considered to be presented
    :param mhc_1_alleles_to_load: list - list of alleles to load from the data_dir_path
    """

    def __init__(self, name, data_dir_path, limit, mhc_1_alleles_to_load=None):
        log_info(f"Init {type(self)} MHC-Class1 predictor")
        self.percentiles = defaultdict(
            lambda: {}
        )  # 2 level dictionary: el_ranks[mhc][peptide] = el_rank
        self.missing = defaultdict(
            lambda: set()
        )  # collect the missing peptides to be requested from the predictor in one go
        self.unavailable = defaultdict(
            lambda: set()
        )  # collect the peptides that could not be predicted
        self.factor = 1
        self.name = name
        self.data_dir_path = data_dir_path
        self.limit = limit
        if data_dir_path is not None:
            self.load_percentiles(
                data_dir_path, mhc_1_alleles_to_load=mhc_1_alleles_to_load
            )
        self.updated_percentiles = False

    @staticmethod
    def get_predictor(class_name):
        """Returns the predictor class specified by the class_name

        Loads the module with the predictor (its name is equal to '_' plus the class name
        without the Mhc1Predictor prefix).
        From this module it returns the class that has the class_name

        :param class_name: str - name of the predictor class
        :return: class - the predictor class
        """

        name_module = class_name.removeprefix("Mhc1Predictor").lower()
        # pylint: disable=unused-variable
        module = importlib.import_module(
            f"._{name_module}", package="kit.bioinf.immuno.mhc_1"
        )
        return eval(f"module.{class_name}")
    
    def predict_peptides(self, peptides, alleles):
        """Calls the underlying prediction algorithm
        and saves the results in the self.percentiles attribute
        Has to be implemented by the child class

        :param peptides: list - list of peptide sequences
        :param allele: str - MHC allele name
        """
        raise NotImplementedError()
    
    def assess_peptide(self, peptide, allele, queue_if_missing=True, predict_if_missing=False):
        """Returns the presentation status as well as the rank of a peptide for the specified MHC allele"""
        raise NotImplementedError()
    
    def get_predictor_hash(self):
        raise NotImplementedError()

    def save(self, percentiles_dir_path=None):
        """Saves the percentiles to the data_dir_path is any changes were made"""

        if self.updated_percentiles:
            percentiles_dir_path = self.data_dir_path if percentiles_dir_path is None else percentiles_dir_path
            self.save_percentiles(percentiles_dir_path)

    def load_percentiles(
        self, data_dir_path, prefix="percentile", mhc_1_alleles_to_load=None
    ):
        """Loads the percentiles from the data_dir_path"""

        for dirname, _, filenames in os.walk(data_dir_path, followlinks=True):
            for filename in filenames:
                h = re.findall(
                    rf"^{prefix}_HLA_([ABCEFGKL])_(\d+)_(\d+)\.tsv$", filename
                )
                if len(h) == 1 and len(h[0]) == 3:
                    mhc = f"HLA-{h[0][0]}*{h[0][1]}:{h[0][2]}"
                    if mhc_1_alleles_to_load is None or mhc in mhc_1_alleles_to_load:
                        log_info(f"Loading {mhc} from {filename}")
                        with open(os.path.join(dirname, filename), "r") as f:
                            perc_file = f.read()
                        perc_file = perc_file.split("\n")
                        for line in perc_file[1:]:
                            line = line.split("\t")
                            if len(line) == 2:
                                peptide, percentile = line
                                self.percentiles[mhc][peptide] = float(percentile)

    def save_percentiles(self, data_dir_path, prefix="percentile"):
        """Saves the percentiles to the data_dir_path"""

        for mhc, mhc_percentiles in self.percentiles.items():
            hla_split = get_mhc_name_split(mhc)
            filename = "_".join([prefix, "HLA"] + list(hla_split))
            filename = join(data_dir_path, f"{filename}.tsv")

            df = pd.DataFrame(
                index=list(mhc_percentiles.keys()),
                columns=[prefix],
                data=list(mhc_percentiles.values()),
            )
            df.index.name = "peptide"
            df.to_csv(filename, sep="\t")

    def queue_peptides(self, peptides, allele):
        """Queues a peptide for prediction later by predict_missing_peptides"""

        for peptide in peptides:
            if not peptide in self.percentiles[allele]:
                self.missing[allele].add(peptide)

    def queue_seq(self, seq, alleles, lengths=None):
        """Queues all peptides in a sequence for prediction later by predict_missing_peptides"""

        if lengths is None:
            lengths = MHC_1_PEPTIDE_LENGTHS

        kmers = get_kmers(seq, lengths)

        alleles = self.resolve_alleles(alleles)
        for allele in alleles:
            self.queue_peptides(kmers, allele)

    def peptide_presented(self, peptide, alleles):
        """Predicts if a peptide is presented by the specified MHC allele

        Calls the assess_peptide method to get the prediction of whether the peptide
        is presented by the allele.

        :param peptide: str - peptide sequence
        :param allele: str - MHC allele name
        :return: bool - True if the peptide is presented by the allele, False if not
            or None if the peptide has not been predicted yet
        """

        result = False
        _alleles = self.resolve_alleles(alleles)
        for allele in _alleles:
            presented, _ = self.assess_peptide(peptide, allele)
            result |= presented
        return result

    def peptide_percentile(self, peptide, allele):
        """Returns the predicted percentile of a peptide for the specified MHC allele

        Will crash if the peptide has not been predicted yet.

        Args: see peptide_presented
        """
        if peptide in self.percentiles[allele]:
            return self.percentiles[allele][peptide] * self.factor
        return None

    def seq_presented(self, seq, alleles, lengths=None, cnt_sep=False):
        """finds all peptides within a sequence that get presented by
        the specified MHC alleles

        :param seq: str - protein sequence
        :param alleles: list - list of MHC allele names
        :param lengths: list - list of peptide lengths to consider
            Defaults to MHC_1_PEPTIDE_LENGTHS
        :param cnt_sep: bool - should the separator (/) between chains be counted for the
            returned peptide terminal indices
        :return: list - list of tuples (peptide, allele, rank,
            sequence position index of the peptide's last AA)
        """

        presented = []

        # in case the sequence has several chains, deal with each individually and then aggregate
        # the separators are then not counted as indices
        if '/' in seq:
            chains = seq.split('/')
            offset = 0
            for chain in chains:
                _presented = self.seq_presented(chain, alleles, lengths=lengths)
                presented += [(_p[0], _p[1], _p[2], _p[3] + offset) for _p in _presented]
                offset += len(chain) + (1 if cnt_sep else 0)
            return presented

        if lengths is None:
            lengths = MHC_1_PEPTIDE_LENGTHS

        seq = seq.replace("*", "").replace("-", "").replace("?", "")

        alleles = self.resolve_alleles(alleles)

        if len(seq) >= min(lengths):
            kmers = get_kmers(seq, lengths)

            for allele in alleles:
                self.queue_peptides(kmers, allele)
                self.predict_missing_peptides()

                for length in lengths:
                    for end in range(length, len(seq)+1):
                        peptide = seq[end - length : end]
                        pres, rank = self.assess_peptide(peptide, allele)
                        if pres is None:
                            presented.append((peptide, allele, None, end - 1))
                        elif pres:
                            presented.append((peptide, allele, rank, end - 1))

        return presented

    def get_presented_positions(self, seq, alleles, lengths=None, anchors=[1, -1]):
        visible = self.seq_presented(seq, alleles, lengths=lengths)

        positions_epitope, positions_anchors, positions_unknown = [], [], []
        for vis in visible:
            first_idx = vis[3] - len(vis[0]) + 1
            term_idx = vis[3]
            if vis[2] is not None:
                positions_epitope += list(range(first_idx, term_idx + 1))
                positions_anchors += [first_idx + p if p > 0 else term_idx + 1 + p for p in anchors]  # [first_idx + 1, term_idx]
            else:
                positions_unknown += list(range(first_idx, term_idx + 1))

        positions_epitope = list(set(positions_epitope))
        positions_anchors = list(set(positions_anchors))
        positions_unknown = list(set(positions_unknown))

        return positions_epitope, positions_anchors, positions_unknown

    def resolve_alleles(self, alleles):
        if "+" in alleles:
            alleles = alleles.split("+")

        if isinstance(alleles, str):
            alleles = [alleles]
        return alleles

    def predict_missing_peptides(self):
        """Predicts all peptides that have been queued for prediction"""

        for alleles, peptides in self.missing.items():
            peptides = list(peptides)
            alleles = self.predict_peptides(peptides, alleles)
            for allele in alleles:
                for peptide in peptides:
                    if peptide not in self.percentiles[allele]:
                        self.unavailable[allele].add(peptide)

        self.missing = defaultdict(lambda: set())

    def get_seq_kmers(self, seq, allele_names, kmer_length):
        """Returns a dataframe with the predicted ranks of all kmers in the sequence

        Each row represents a kmer in the sequence
        There is a column for each allele in allele_names. The value in the column
            represents the rank of the kmer for the allele.
        There are two additional columns:
            visibility - number of alleles for which the kmer is presented
            presented - True if the kmer is presented by at least one allele
        """

        kmers = seq_to_kmers_list(seq, kmer_length)
        df = pd.DataFrame(index=kmers)

        for row in df.iterrows():
            visibility = 0
            for allele_name in allele_names:
                if allele_name not in df.columns:
                    df[allele_name] = None
                presented, percentile = self.assess_peptide(row.name, allele_name)
                visibility += 1 if presented else 0
                df.loc[row.name, allele_name] = percentile

            if "visibility" not in df.columns:
                df["visibility"] = None
            if "presented" not in df.columns:
                df["presented"] = None

            df.loc[row.name, "visibility"] = visibility
            df.loc[row.name, "presented"] = visibility > 0

        df.seq_hash = get_seq_hash(seq)
        return df
