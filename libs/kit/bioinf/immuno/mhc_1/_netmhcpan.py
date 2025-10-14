"""This module conatains the Mhc1PredictorNetMhcPan class.
as well as functions to interact with the netMHCpan cli."""

import os
import re
import subprocess
import pandas as pd

from kit.data import file_to_str, str_to_file
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.utils import temp_working_directory
from kit.hashes import str_to_hash

from kit.bioinf.immuno.utils import get_mhc_name_split, get_mhc_class


class Mhc1PredictorNetMhcPan(Mhc1Predictor):
    """A class for predicting MHC class 1 presentation percentiles using netMHCpan

    :param data_dir_path: str - path to the directory containing the percentile files
        This avoids predicting the same pMHC multiple times and so speeds
        up the prediction process.
    :param limit: float - percentile limit below which a peptide is considered presented
    :param mhc_1_alleles_to_load: list - list of alleles to load the percentile files
        for. If None, all alleles are loaded.
    """

    resolved_alleles = {}

    def __init__(self, data_dir_path=None, limit=0.02, mhc_1_alleles_to_load=None):
        super().__init__(
            "netMHCpan",
            data_dir_path,
            limit,
            mhc_1_alleles_to_load=mhc_1_alleles_to_load,
        )
        self.factor = 0.01

    def get_predictor_hash(self):
        return str_to_hash(f"{self.name} {self.limit}")

    def predict_peptides(self, peptides, alleles):
        """Predicts the predicted presentation percentiles of the peptides for the allele."""

        _alleles = self.resolve_alleles(alleles)
        for _allele in _alleles:
            with temp_working_directory() as _:
                str_to_file("\n".join(peptides), "temporary.pep")
                result = subprocess.run(
                    [
                        "netMHCpan",
                        "-p temporary.pep",
                        f"-a {to_netmhcpan_allele_name(_allele)}",
                    ],
                    capture_output=True,
                    check=False,
                )
                interpret_netmhcpan_output(
                    result.stdout.decode("UTF-8").split("\n"), self.percentiles
                )
                self.updated_percentiles = True
                
        return _alleles

    def assess_peptide(self, peptide, allele):
        rank = self.peptide_percentile(peptide, allele)
        if rank is None:
            return None, None
        return rank < self.limit, rank

    def resolve_alleles(self, alleles):
        if "+" in alleles:
            alleles = alleles.split("+")

        if isinstance(alleles, str):
            return self.resolve_allele(alleles)
        
        result = []
        for allele in alleles:
            result += self.resolve_alleles(allele)
        return result
    
    def resolve_allele(self, allele):
        if allele in Mhc1PredictorNetMhcPan.resolved_alleles:
            return Mhc1PredictorNetMhcPan.resolved_alleles[allele]
        
        if self.data_dir_path is not None \
         and os.path.exists(definition_file_path := os.path.join(self.data_dir_path, "definitions", f"{allele}_definition.csv")):
            df = pd.read_csv(definition_file_path)
            result = [f"HLA-{a}" for a in df.allele]
        else:
            result = [allele]    

        Mhc1PredictorNetMhcPan.resolved_alleles[allele] = result
        return result


def interpret_netmhcpan_output(lines, el_ranks):
    """Interprets the output of netMHCpan cli"""

    line_nr = 1
    allele = None
    for line in lines:
        if line_nr == 42:
            allele = f"{line[:5]}*{line[5:10]}"

        if line_nr >= 49 and line[5:8] == "HLA":
            regex = (
                r"\s+\d+\s+HLA-[ABCEFGKL]\*\d+:\d+\s+(\w+)\s+(?:\w|-)+"
                + r"\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\w+\s+PEPLIST\s+([\d.]+)\s+([\d.]+)"
            )
            peptide, _, rank_el = re.findall(regex, line)[0]
            if peptide.isnumeric():
                raise ValueError("netMHCpanPredictor: Peptide is numeric")
            el_ranks[allele][peptide] = float(rank_el)

        line_nr += 1


def to_netmhcpan_allele_name(allele_name):
    """Converts an allele name to the format used by netMHCpan"""

    if get_mhc_class(allele_name) == 1:
        h = get_mhc_name_split(allele_name)
        return f"HLA-{h[0]}{h[1]}:{h[2]}"
    raise ValueError(f"Invalid allele name: {allele_name}")
