"""MHC1 predictor based on PWMs."""

import os
import numpy as np
import pandas as pd

from kit.path import join
from kit.maths import get_precision, get_recall
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.hashes import str_to_hash


class Mhc1PredictorPwm(Mhc1Predictor):
    """Fast MHC1 predictor based on PWMs.

    :param data_dir_path: Path to the folder with the PWMs.
    :param limit: Percentile limit below which a peptide is considered presented.
    :param aa_na: Score for amino acids not present in the PWM.
        The score of a peptide is the sum of the scores of its amino acids.
        Each score is the log likelihood of the amino acid at the given position
            in presented peptides of the allele.
        The percentile of a peptide is the percentile of its score among all peptides
            This is approximated via linear interpolation
    """

    def __init__(self, data_dir_path=None, limit=0.02, limit_calibration=None, aa_na=0):
        if os.path.exists(os.path.join(data_dir_path, "pwm")):
            self.pwm_dir_path = os.path.join(data_dir_path, "pwm")
        else:
            self.pwm_dir_path = data_dir_path
        super().__init__("PWM", data_dir_path, limit)
        self.factor = 0.01
        self.PWMs_log = {}     # PWMs_log[allele][length][aa_idx, pos] = log probability  (numpy array)
        self.info_pc = {}
        self.score_limits = {}
        self.pc_limits = {}
        self.aa_na = aa_na
        self.aa_idx = None
        self.limit_calibration = limit_calibration if limit_calibration else limit

        # only used during new PWMs creation
        self.PWMs = {}          # PWMs[allele][length].loc[aa, pos] = probability  (DataFrame)
        self.ranks = {}         # ranks[allele].loc[peptide] = rank
        self.presented_any = {}     # presented[allele].loc[peptide] = presented

    def get_predictor_hash(self):
        return str_to_hash(f"{self.name} {self.limit} {self.limit_calibration} {self.aa_na}")

    def save(self):
        """no information should be saved for this predictor"""
        pass

    def predict_peptides(self, peptides, alleles):
        """Predicts the percentiles of the peptides for the allele."""

        _alleles = self.resolve_alleles(alleles)
        for allele in _alleles:
            for peptide in peptides:
                self.predict_peptide(peptide, allele)

        return _alleles

    def load_allele(self, allele, length):
        """Loads the PWMs for the allele and length.

        :param allele: Allele name.
        :param length: Peptide length.
        """

        if allele not in self.PWMs_log:
            self.PWMs_log[allele] = {}
            self.info_pc[allele] = {}
            self.score_limits[allele] = {}
            self.pc_limits[allele] = {}

        if length not in self.PWMs_log[allele]:
            allele_for_path = allele.replace("*", "_")
            self.PWMs_log[allele][length] = (
                pd.read_csv(
                    os.path.join(
                        self.pwm_dir_path,
                        allele_for_path,
                        f"{allele_for_path}-{length}_log.csv",
                    )
                )
                .set_index("AA")
                .sort_index()
            )
            self.PWMs_log[allele][length].columns = [
                int(c) for c in self.PWMs_log[allele][length].columns
            ]
            self.set_aa_idx(self.PWMs_log[allele][length].index)

            stats_file_path = os.path.join(
                self.pwm_dir_path,
                allele_for_path,
                f"stats-{allele.replace('*', '_')}-{self.limit_calibration}-{length}.csv",
            )
            if not os.path.exists(stats_file_path):
                old_stats_file_path = os.path.join(
                    self.pwm_dir_path,
                    allele_for_path,
                    f"pc-{allele.replace('*', '_')}-{self.limit_calibration}-{length}.csv",
                )
                info = pd.read_csv(old_stats_file_path).rename(columns={allele: "score"}).set_index("Info")
                info = info.loc[info.index.str.startswith("pc_")]
                info["pc"] = info.apply(lambda row: float(row.name[3:]), axis=1)
                info = info[["pc", "score"]].set_index("pc").sort_index(ascending=True)
            else:
                info = pd.read_csv(stats_file_path).set_index("percentile").sort_index(ascending=True)

            y = [-2e9] + list(info["score"].to_numpy()) + [0.0]
            x = [0.0] + list(info.index) + [100.0]
            if isinstance(self.limit, float):
                _limit = self.limit
            elif self.limit == "dynamic":
                row = info.iloc[0]
                _presented = (row.TP + row.FN)
                _total = _presented + (row.TN + row.FP)
                _limit = _presented / _total
            else:
                self.PWMs_log[allele] = {}
                raise ValueError(f"Unknown limit type: {self.limit}")
            
            self.score_limits[allele][length] = np.interp(100.0 - 100.0 * _limit, x, y)
            self.pc_limits[allele][length] = _limit
            self.info_pc[allele][length] = info
            self.PWMs_log[allele][length] = self.PWMs_log[allele][length].values 

    def set_aa_idx(self, aa_list):
        if self.aa_idx is None:
            self.aa_idx = {}
            for idx, aa in enumerate(aa_list):
                self.aa_idx[aa] = idx
        else:
            for idx, aa in enumerate(aa_list):
                assert self.aa_idx[aa] == idx

    def peptide_presented(self, peptide, allele):
        """Returns whether the peptide is predicted to be presented by the allele."""

        length = len(peptide)
        score = self.score_peptide(peptide, allele)
        return score > self.score_limits[allele][length]

    def predict_peptide(self, peptide, allele):
        """Predicts the percentile of the peptide for the allele."""

        length = len(peptide)
        self.load_allele(allele, length)

        score = self.score_peptide(peptide, allele)

        x = [-2e9] + list(self.info_pc[allele][length]['score'].to_numpy()) + [0.0]
        y = [0.0] + list(self.info_pc[allele][length].index) + [100.0]

        self.updated_percentiles = True
        self.percentiles[allele][peptide] = 100. - np.interp(score, x, y)

    def assess_peptide(self, peptide, allele):
        pc = self.peptide_percentile(peptide, allele)
        if pc is None:
            return None, None
        
        length = len(peptide)
        return pc < self.pc_limits[allele][length], pc

    def score_peptide(self, peptide, allele):
        """Returns the score of the peptide for the allele."""

        length = len(peptide)
        pwm_log = self.PWMs_log[allele][length]
        score = 0.0
        for pos_c, aa in enumerate(peptide):
            pos_r = self.aa_idx.get(aa, -1)
            if pos_r >= 0:
                score += pwm_log[pos_r, pos_c]
            else:
                score += self.aa_na
        return score

    def score_peptides(self, peptides, allele):
        scores = []
        for peptide in peptides:
            scores.append(self.score_peptide(peptide, allele))

        return scores
    
    #
    #
    # The followin methods are used to create the PWMs
    #
    #

    def get_allele_ranks(self, allele, load=True):
        """Returns the data frame with the ranks of the peptides for the allele
        If not present, returns None."""

        if allele.startswith("HLA-"):
            allele = allele[4:]

        csv_file_path = os.path.join(
            self.data_dir_path, 
            "ranks",
            f"HLA-{allele.replace('*', '_')}.csv"
        )
        
        if not os.path.exists(csv_file_path):
            ranks = None
        else:
            ranks = pd.read_csv(csv_file_path).drop_duplicates().set_index('peptide')
        
        if load:
            self.ranks[allele] = ranks

        return ranks
        
    def load_agg_ranks(self, agg_name, alleles, save_definition=True):
        """Returns a data frame with the ranks of the random peptides used for calibration.
        Each column is the rank of the peptide for the allele.
        Missing alleles are returned in a separate list.
        
        The method also saves the definition of the alleles in the data directory.
        
        :param agg_name: Name of the aggregation.
        :param alleles: List of alleles."""

        df_ranks, peptides, missing_alleles = None, None, []
        for allele in alleles:
            df = self.get_allele_ranks(allele, load=False)
            if df is not None:
                if peptides is None:
                    peptides = set(df.index)
                else:
                    if len(peptides^set(df.index)) > 0:
                        raise ValueError(f"Peptides for allele {allele} are not the same as for the other alleles")    
                    
                if df_ranks is None:
                    df_ranks = df
                else:
                    df_ranks = df_ranks.join(df, how="inner")
            else:
                missing_alleles.append(allele)

        self.ranks[agg_name] = df_ranks

        if save_definition:
            df_definition = pd.DataFrame(alleles, columns=["allele"]).set_index("allele")
            df_definition['missing'] = df_definition.index.isin(missing_alleles)
            df_definition.to_csv(join(self.data_dir_path, "definitions", f"{agg_name}_definition.csv"))

        return missing_alleles
    
    def get_presented_any(self, name, recalculate=False):
        """Returns a list of presented peptides
        A peptide is considered presented if it is in the top limit_calibration percentile of any allele.
        
        :param name: Name of the aggregation or allele
        :param recalculate: If True, recalculates the presented peptides also if already done before. Default is False.
        """

        if recalculate or name not in self.presented_any:
            df_ranks = self.ranks.get(name, None)
            if df_ranks is None:
                raise ValueError(f"Ranks for {name} are not available")
            
            self.presented_any[name] = pd.DataFrame(
                (df_ranks < self.limit_calibration).apply(any, axis=1),
                columns=['presented']
            )

        return list(self.presented_any[name].query('presented').index)

    #
    # START: create PWMs
    #
    def calc_PWMs(self, allele, counts):
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

            # check that the aa are in the right order
            self.set_aa_idx(PWMs_log[length].index)

        self.PWMs[allele] = PWMs
        for length in PWMs_log.keys():
            PWMs_log[length] = PWMs_log[length].values
        self.PWMs_log[allele] = PWMs_log
    
    def save_PWMs(self, allele):
        """Saves the PWMs for the allele."""
        PWMs = self.PWMs[allele]
        PWMs_log = {}
        for length in self.PWMs_log[allele].keys():
            PWMs_log[length] = pd.DataFrame(
                index=PWMs[length].index, 
                columns=PWMs[length].columns, 
                data=self.PWMs_log[allele][length]
            )

        lengths = set(PWMs.keys())
        if len(lengths ^ set(PWMs_log.keys())) > 0:
            raise ValueError("Lengths for PWMs and PWMs_log are not the same")

        for_path = allele.replace("*", "_")

        for length in lengths:
            # save the produced PWMs
            PWMs[length].to_csv(
                join(
                    self.data_dir_path,
                    "pwm",
                    for_path,
                    f"{for_path}-{length}.csv",
                )
            )
            PWMs_log[length].to_csv(
                join(
                    self.data_dir_path,
                    "pwm",
                    for_path,
                    f"{for_path}-{length}_log.csv",
                )
            )    

    #
    # END: create PWMs
    # START: Statistics
    #
    def calc_percentile_statistics(self, name):
        self.get_presented_any(name)
        df_presented = self.presented_any[name]

        percentile_statistics = {}
        if 'length' not in df_presented.columns:
            df_presented['length'] = df_presented.apply(lambda x: len(x.name), axis=1)

        # percentiles = np.linspace(100., 0., 201)
        # percentiles[0] = 99.9
        # percentiles[-1] = 0.1

        percentiles = np.concatenate([
            np.array([99.99]),
            np.linspace(99.9, 99.1, 9), 
            np.linspace(99., 95.25, 16), 
            np.linspace(95., 0.5, 190),
            np.array([0.1]),
        ])

        scores = self.score_peptides(list(df_presented.index), name)
        df_presented['score'] = scores

        for length in df_presented.length.unique():
            df_positives = df_presented.query(f"length == {length}").copy()
            df = pd.DataFrame(
                index=percentiles,
                columns=["score", "TP", "FP", "TN", "FN"],
            )
            df.index.name = "percentile"

            df['score'] = np.percentile(
                df_positives["score"],
                percentiles,
            )

            for pc in percentiles:
                threshold = df.loc[pc, 'score']
                df_positives['positive'] = df_positives.apply(
                    lambda x: x['score'] >= threshold,
                    axis=1,
                )
                TOTALS = df_positives.shape[0]
                TP = df_positives.query("presented and positive").shape[0]
                FP = df_positives.query("not presented and positive").shape[0]
                TN = df_positives.query("not presented and not positive").shape[0]
                FN = df_positives.query("presented and not positive").shape[0]
                assert(TP + FP + TN + FN == TOTALS)

                df.loc[pc, 'precision'] = get_precision(TP, FP)
                df.loc[pc, 'recall'] = get_recall(TP, FN)

                df.loc[pc, 'TP'] = TP
                df.loc[pc, 'FP'] = FP
                df.loc[pc, 'TN'] = TN
                df.loc[pc, 'FN'] = FN

            percentile_statistics[length] = df

        return percentile_statistics

    def save_percentile_statistics(self, allele_or_name, percentile_statistics):
        folder = self.data_dir_path
        for_path = allele_or_name.replace("*", "_")
        for length, df in percentile_statistics.items():
            df.to_csv(
                join(
                    folder,
                    "pwm",
                    for_path,
                    f"stats-{for_path}-{self.limit_calibration}-{length}.csv",
                )
            )

    #
    # END: Statistics
    #
        