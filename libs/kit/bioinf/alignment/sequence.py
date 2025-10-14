"""Holds classes for sequence alignment."""

import os
import subprocess

import numpy as np

from Bio import Align, AlignIO
from Bio.Align import substitution_matrices
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

from kit.log import log_info
from kit.bioinf.sf import SequenceFrame
from kit.utils import temp_working_directory


class PairwiseSequenceAligner:
    """Class to align sequences using the Biopython pairwise aligner.

    :param substitution_matrix: str - name of the substitution matrix
    :param open_gap_score: int - score penalty for opening a gap
    :param extend_gap_score: int - score penalty for extending a gap
    :param wildcard: str - wildcard character (will be ignored)
    """

    def __init__(
        self,
        substitution_matrix="BLOSUM62",
        open_gap_score=-2,
        extend_gap_score=-1,
        wildcard="-",
    ):
        self.aligner = (
            Align.PairwiseAligner()
        )  # https://biopython.org/docs/1.75/api/Bio.Align.html#Bio.Align.PairwiseAligner
        self.aligner.substitution_matrix = substitution_matrices.load(
            substitution_matrix
        )
        self.aligner.open_gap_score = open_gap_score
        self.aligner.extend_gap_score = extend_gap_score
        self.aligner.wildcard = wildcard

    def score_seq_to_seq(self, seq1, seq2, translate=("", "", "-*")):
        """Score two sequences using the Biopython pairwise aligner.

        :param seq1: str - first sequence
        :param seq2: str - second sequence
        :param translate: tuple - translation table for the sequences
            Defaults to ("", "", "-*") which means that '-' and '*' will be ignored.
        :return: int - score
        """
        seq1 = seq1.translate(str.maketrans(*translate))
        seq2 = seq2.translate(str.maketrans(*translate))
        return self.aligner.score(seq1, seq2)

    def score_seq_to_seqs(self, seq, tgts, translate=("", "", "-*")):
        """Score a sequence to a list of sequences using the Biopython pairwise aligner.

        :param seq: str - sequence
        :param tgts: list - list of sequences to score seq against
        :param translate: tuple - translation table for the sequences (see score_seq_to_seq)
        :return: np.array - scores
        """
        result = []
        for tgt in tgts:
            s = self.score_seq_to_seq(seq, tgt, translate=translate)
            result.append(s)
        return np.array(result)

    def get_seq_to_seqs_dissimilarity(self, seq, seqs):
        """Get the dissimilarity between a sequence and a list of sequences.

        We define the dissimilarity as the difference between the alignment score of the
        sequence against itself and the alignment score of the sequence against
        the other sequence.

        :param seq: str - sequence
        :param seqs: list - list of sequences to score seq against
        :return: np.array - dissimilarities
        """
        max_score = self.score_seq_to_seq(seq, seq)
        scores = self.score_seq_to_seqs(seq, seqs)
        return max_score - scores


class MultipleSequenceAligner:
    """Class to perform mulitple sequence alignment using MUSCLE.

    It also supports the calculation of phylogenetic trees

    :param sf: SequenceFrame - sequences to align
    :param distance_model_name: str - name of the distance model to use
        Defaults to 'blosum62'.
    :param phylogenetic_tree_model_name: str - name of the phylogenetic tree model to use
        - 'nj'
        - 'upgma'
        - 'raxml'

    """

    def __init__(
        self, sf, distance_model_name="blosum62", phylogenetic_tree_model_name="nj"
    ):
        self.sf = sf
        self.msa_file_path = None
        self.msa = None
        self.occupancy = None
        self.occupancy_threshold = None
        self.occupancy_indices_above_threshold = None
        self.distance_model_name = distance_model_name
        self.distance_matrix = None
        self.phylogenetic_tree_model_name = phylogenetic_tree_model_name

    def align(self, msa_file_path=None):
        """Align sequences in self.sf using MUSCLE.

        Args:
            msa_file_path (str): path to the output MSA file.
                If None, the MSA file will be saved in a temporary directory.
                If the filename ends with '.phy', the MSA file will be saved
                    in PHYLIP format otherwise in FASTA format.
        """

        self.msa_file_path = msa_file_path
        with temp_working_directory() as tmp_dir_path:
            input_file_path = os.path.join(tmp_dir_path, "input.fa")
            output_file_path = (
                os.path.join(tmp_dir_path, "output.afa")
                if msa_file_path is None
                else msa_file_path
            )

            self.sf.to_fasta(input_file_path, save_src=False)
            mode = "-super5" if len(self.sf) > 500 else "-align"
            command = ["muscle", mode, input_file_path, "-output", output_file_path]

            log_info(f"Muscle: {' '.join(command)}", start=True)
            # pylint: disable=unused-variable
            result = subprocess.run(command, capture_output=True, check=False)
            log_info("Muscle finished", stop=True)
            self.msa = AlignIO.read(output_file_path, "fasta")
            self.distance_matrix = None

            sf_msa = SequenceFrame()
            if output_file_path.endswith(".phy"):
                AlignIO.write(self.msa, output_file_path, "phylip-relaxed")
                sf_msa.from_phy(output_file_path)
            else:
                sf_msa.from_fasta(output_file_path)
            assert len(self.sf) == len(sf_msa)

            self.sf.df_src["seq_msa"] = ""
            for (_, row_pre), (_, row_post) in zip(
                self.sf.df.iterrows(), sf_msa.df_src.iterrows()
            ):
                seq_msa = row_post.seq_src
                seq_hash = row_pre.seq_hash

                indices = row_pre["index"]
                for index in indices:
                    assert seq_hash == self.sf.df_src.at[index, "seq_hash"]
                    self.sf.df_src.at[index, "seq_msa"] = seq_msa

            self.sf.set_view()

    def construct_phylogenetic_tree(self, method="ascii", do_show=True):
        """Construct a phylogenetic tree

        If the phylogenetic tree model name is not 'raxml',
        the tree will be printed as characters (method == 'ascii') or as a
        plot (method == 'plot').

        :params method: str - 'ascii' or 'plot'
        """
        if self.msa is not None:
            if (
                self.distance_matrix is None
                and self.phylogenetic_tree_model_name != "raxml"
            ):
                log_info(f"Calculating distances using {self.distance_model_name}")
                calculator = DistanceCalculator(self.distance_model_name)
                self.distance_matrix = calculator.get_distance(self.msa)
                log_info("Finished calculating distances")

            tree = None
            constructor = DistanceTreeConstructor()
            if self.phylogenetic_tree_model_name == "nj":
                tree = constructor.nj(self.distance_matrix)
            elif self.phylogenetic_tree_model_name == "upgma":
                tree = constructor.upgma(self.distance_matrix)
            elif self.phylogenetic_tree_model_name == "raxml":
                if self.msa_file_path is None or not self.msa_file_path.endswith(
                    ".phy"
                ):
                    raise ValueError(
                        "a '.phy' file must be specified as 'msa_file_path' in 'align' for raxml"
                    )

                output_dir_path = os.path.dirname(self.msa_file_path)
                with temp_working_directory(output_dir_path) as _:
                    # Remove files starting with 'RAxML_' in the output directory
                    directory_path = os.path.dirname(self.msa_file_path)
                    file_list = os.listdir(directory_path)
                    for file_name in file_list:
                        if file_name.startswith("RAxML_"):
                            file_path = os.path.join(directory_path, file_name)
                            os.remove(file_path)

                    # raxmlHPC -s output.phy -n my_tree -m PROTGAMMAGTR -p 42
                    command = [
                        "raxmlHPC",
                        "-s",
                        os.path.basename(self.msa_file_path),
                        "-n",
                        "my_tree",
                        "-m",
                        "PROTGAMMAGTR",
                        "-p",
                        "42",
                    ]
                    log_info(f"Raxml: {' '.join(command)}", start=True)
                    # pylint: disable=unused-variable
                    result = subprocess.run(command, capture_output=True, check=False)
                    log_info("Raxml finished", stop=True)
            else:
                raise ValueError(
                    f"Unknown phylogenetic tree model name: {self.phylogenetic_tree_model_name}"
                )

            if self.phylogenetic_tree_model_name != "raxml" and do_show:
                if method == "ascii":
                    Phylo.draw_ascii(tree)
                elif method == "plot":
                    Phylo.draw(tree)
            return tree
        raise ValueError("MSA is not calculated yet")

    def calculate_occupancy(self):
        """For all positions in the MSA, calculates how many sequences have
        no gap at that position."""

        if self.msa is not None:
            alignment_length = self.msa.get_alignment_length()

            self.occupancy = []
            for i in range(alignment_length):
                column = self.msa[:, i]
                self.occupancy.append(1.0 - column.count("-") / len(column))
        else:
            raise ValueError("MSA is not calculated yet")

    def set_occupancy_threshold(self, threshold):
        """Set the occupancy threshold for the MSA.

        Finds all positions in the MSA where the occupancy is above the threshold
        Adds a new column to the SequenceFrame (df_src) with the sequences where only the
        positions above the threshold are kept.

        :param threshold: float - threshold for the occupancy
        """
        if self.occupancy is None:
            self.calculate_occupancy()
        self.occupancy_threshold = threshold

        self.occupancy_indices_above_threshold = []
        for idx, value in enumerate(self.occupancy):
            if value >= threshold:
                self.occupancy_indices_above_threshold.append(idx)

        self.sf.df_src[f"seq_msa_{threshold}"] = self.sf.df_src.seq_msa.apply(
            lambda seq_msa: "".join(
                [seq_msa[idx] for idx in self.occupancy_indices_above_threshold]
            )
        )
        self.sf.set_view()

    @staticmethod
    def seq_idx_to_msa_idx(seq_msa):
        """Produces a list that specifies for each position in the sequence
        which position it has in the MSA

        :param seq_msa: str - sequence in the MSA (including '-')
        :return: msa_idx_list - list of indices in the MSA
        """

        msa_idx_list = []
        msa_idx = 0
        while msa_idx < len(seq_msa):
            if seq_msa[msa_idx] != "-":
                msa_idx_list.append(msa_idx)
            msa_idx += 1

        return msa_idx_list

    @staticmethod
    def msa_idx_to_seq_idx(seq_msa):
        """Based on a sequence, specifies for each position in the MSA
        which position it has in the sequence. If the position does not exist
        in the sequence (because it is a gap), None is used.

        :param seq_msa: str - sequence in the MSA (including '-')
        :return: idx_list - list of indices in the sequence
        """
        idx_list = []
        msa_idx = 0
        idx = 0
        while msa_idx < len(seq_msa):
            if seq_msa[msa_idx] == "-":
                idx_list.append(None)
            else:
                idx_list.append(idx)
                idx += 1
            msa_idx += 1

        return idx_list
