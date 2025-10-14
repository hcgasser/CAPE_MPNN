import os

from enum import Enum
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB import PDBParser, PPBuilder
import numpy as np
import pandas as pd

from kit.log import log_info
from kit.path import join
from kit.bioinf.fasta import seqs_to_fasta
from kit.bioinf.pdb import structure_to_pdb
from kit.bioinf.utils import keep_chains_in_structure, structure_to_seq, get_seq_hash, chains_to_seq


destress_infos = {
    # info_attr: csv column name
    'dssp': 'dssp_assignment',
    'rosetta_total': 'rosetta_total',
    'hydrophobic_fitness': 'hydrophobic_fitness',
    'packing_density': 'packing_density',
    'aggrescan3d_avg': 'aggrescan3d_avg_value',
    'aggrescan3d_max': 'aggrescan3d_max_value',
    'dfire2_total': 'dfire2_total',
    'evoef2_total': 'evoef2_total',
    'isoelectric_point': 'isoelectric_point',
    'charge': 'charge',
    'mass': 'mass',
}


class Protein:
    proteins = {}
    loch = None

    def __init__(self):
        self.seq_hash = None

        self.chains = None
        self.structure = None
        self.phi_psi = None
        self.only_polypeptides = True

        # destress
        for info_attr in destress_infos.keys():
            setattr(self, info_attr, None)

        self.ref_seq_hash = None

    @classmethod
    def from_loch(cls, seq_hash, predictor_structure_name='AF', register=True):
        pdb_file_path = Protein.loch.get_pdb_file_path(seq_hash, predictor_structure_name=predictor_structure_name)
        Protein.from_pdb(pdb_file_path, seq_hash=seq_hash, register=register)

    @classmethod
    def from_pdb(cls, pdb_file_path, seq_hash=None, register=True, keep_chains=None, 
            structure_to_seq_kwargs=None, seq_translate=('', '', '')):
        protein = Protein()

        pdb_parser = PDBParser(QUIET=True)
        protein.structure = pdb_parser.get_structure('protein', pdb_file_path)
        keep_chains_in_structure(protein.structure, keep_chains)
        if structure_to_seq_kwargs is not None:
            seq = structure_to_seq(protein.structure, **structure_to_seq_kwargs)
        else:
            seq = structure_to_seq(protein.structure)

        if len(seq) > 1:
            log_info(f"Multiple models found in {pdb_file_path}. Using the first one.")
        elif len(seq) == 0:
            raise Exception(f"No sequence found in {pdb_file_path}")
        protein.chains = seq[0]
        translate_table = str.maketrans(*seq_translate)
        for c_id, chain in protein.chains.items():
            protein.chains[c_id] = chain.translate(translate_table)

        protein.phi_psi = calc_phi_psi(protein.structure)

        protein.seq_hash = seq_hash if seq_hash is not None else get_seq_hash(protein.chains)

        if register:
            Protein.proteins[protein.seq_hash] = protein
        return protein

    def to_loch(self, predictor_structure_name):
        if Protein.loch is None:
            raise Exception("Protein.loch is None")
        if self.structure is None:
            raise Exception("Protein.structure is None")

        pdb_file_path = Protein.loch.get_pdb_file_path(self.seq_hash, predictor_structure_name=predictor_structure_name)
        fasta_file_path = Protein.loch.get_fasta_file_path(self.seq_hash)

        seqs_to_fasta(chains_to_seq(self.chains), fasta_file_path)
        structure_to_pdb(self.structure, pdb_file_path)
        

    def load_destress(self, predictor_structure_name='AF'):
        if self.seq_hash is None or Protein.loch is None:
            raise Exception("seq_hash is None")

        destress_file_path = Protein.loch.get_destress_file_path(self.seq_hash, predictor_structure_name=predictor_structure_name)
        if not os.path.exists(destress_file_path):
            raise Exception(f"destress file not found: {destress_file_path}")

        df_destress = pd.read_csv(destress_file_path)
        for info_attr, column in destress_infos.items():
            setattr(self, info_attr, df_destress.iloc[0][column])        


    def set_info(self, info_attr, value):
        setattr(self, info_attr, value)

    def get_seq(self):
        return chains_to_seq(self.chains)
    
    def get_info(self, info_attr, delta=False, relative=False):
        if hasattr(self, info_attr):
            if delta:
                if self.ref_seq_hash is None:
                    result = None
                else:
                    ref_protein = Protein.proteins[self.ref_seq_hash]
                    result = getattr(self, info_attr) - getattr(ref_protein, info_attr)
            else:
                result = getattr(self, info_attr)
            
            if relative:
                result = result / getattr(self, info_attr)

            return result
        else:
            raise Exception(f"info_attr not found: {info_attr}")

    def get_protein_type(self):
        if self.chains is None:
            return None
        if len(self.chains) == 1:
            return ProteinType.MONOMER
        else:
            if len(set(self.chains.values())) == 1:
                return ProteinType.HOMOOLIGOMER
            elif self.only_polypeptides:
                return ProteinType.HETEROOLIGOMER
            return ProteinType.COMPLEX

    def plot_ramachandran(self):
        assert self.phi_psi is not None
        phi = [np.degrees(angle[0]) for angle in self.phi_psi if angle[0] is not None]
        psi = [np.degrees(angle[1]) for angle in self.phi_psi if angle[1] is not None]

        plt.figure()
        plt.scatter(phi, psi, marker='.', color='blue')
        plt.xlabel('Phi')
        plt.ylabel('Psi')
        plt.title('Ramachandran Plot')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlim(-180, 180)
        plt.ylim(-180, 180)
        plt.show()


def calc_phi_psi(structure):
    phi_psi = []
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        phi_psi.extend(pp.get_phi_psi_list())
    return phi_psi


class ProteinType(Enum):
    MONOMER = 0
    HOMOOLIGOMER = 1
    COMPLEX = 2
    HETEROOLIGOMER = 3

    def pdb_file_path(self, pdb_dir_path, pdb_id=None, ckpt_id=None):
        sub_dir = ""
        if self == self.MONOMER:
            sub_dir = "monomers"
        elif self == self.HOMOOLIGOMER:
            sub_dir = "homooligomers"
        elif self == self.COMPLEX:
            sub_dir = "complexes"
        elif self == self.HETEROOLIGOMER:
            sub_dir = "heterooligomers"
        type_dir_path = join(pdb_dir_path, sub_dir)
        ckpt_dir_path = join(type_dir_path, ckpt_id) if ckpt_id is not None else type_dir_path
        return join(ckpt_dir_path, f"{pdb_id}.pdb") if pdb_id is not None else ckpt_dir_path

    def __str__(self):
        return self.to_text()

    def to_text(self, singular=True):
        if self == self.MONOMER:
            return "monomer" if singular else "monomers"
        elif self == self.HOMOOLIGOMER:
            return "homooligomer" if singular else "homooligomers"
        elif self == self.COMPLEX:
            return "complex" if singular else "complexes"
        elif self == self.HETEROOLIGOMER:
            return "heterooligomer" if singular else "heterooligomers"
