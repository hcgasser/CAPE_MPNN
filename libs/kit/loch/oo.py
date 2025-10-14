import os
import shutil
import subprocess

import pandas as pd

from kit.bioinf.utils import get_seq_hash, chains_to_seq
from kit.loch.path import get_fasta_file_path, get_pdb_file_path, \
    get_destress_file_path, get_function_path, get_md_path, cp_pdb_to_dir
from kit.bioinf.pdb import pdb_to_seqs, download_structure
from kit.bioinf.fasta import fastas_to_seqs
from kit.utils import temp_working_directory

from kit.loch.update import add_seq, rm_seq


PREDICTOR_STRUCTURE_NAMES = ["AF", "exp"]


class Loch:
    def __init__(self, loch_path=None):
        self.loch_path = loch_path
        self.df_pdb_id_to_seq_hash = None

    def add_seq(self, seq):
        return add_seq(seq, loch_path=self.loch_path)
    
    def add_structure(self, seq_hash, pdb_file_path, predictor_structure_name):
        loch_pdb_file_path = get_pdb_file_path(seq_hash, predictor_structure_name=predictor_structure_name, loch_path=self.loch_path)
        if not os.path.exists(loch_pdb_file_path):
            shutil.copy(pdb_file_path, loch_pdb_file_path)
        return loch_pdb_file_path

    def add_entry(self, seq=None, 
                  pdb_id=None,
                  pdb_dir_path=None,
                  pdb_file_path=None, 
                  model_nr=0, 
                  predictor_structure_name='exp', 
                  pdb_to_seqs_kwargs=None,
                  server=r'https://files.rcsb.org'
        ):
        if seq is not None:
            seq_hash = add_seq(seq, loch_path=self.loch_path)

        if pdb_id is not None and pdb_dir_path is not None:
            pdb_file_path = download_structure(pdb_id, pdb_dir_path, server=server)

        if pdb_file_path is not None:
            if pdb_to_seqs_kwargs is not None:
                models = pdb_to_seqs(pdb_file_path, return_full=True, **pdb_to_seqs_kwargs)
            else:
                models = pdb_to_seqs(pdb_file_path, return_full=True)
            chains = models[model_nr]
            pdb_seq = chains_to_seq(chains)
            if seq is not None:
                assert seq == pdb_seq or seq == pdb_seq.replace("-", "X")
            seq = pdb_seq.replace("-", "X")
            seq_hash = get_seq_hash(seq)
            loch_pdb_file_path = get_pdb_file_path(seq_hash, predictor_structure_name=predictor_structure_name, loch_path=self.loch_path)
            if not os.path.exists(loch_pdb_file_path):
                shutil.copy(pdb_file_path, loch_pdb_file_path)

            if pdb_id is not None:
                self.set_pdb_id_to_seq_hash(pdb_id, seq_hash)

        return seq_hash

    def rm_entry(self, seq_hash):
        rm_seq(seq_hash, loch_path=self.loch_path)

    def get_seq(self, seq_hash):
        fasta_file_path = get_fasta_file_path(seq_hash, loch_path=self.loch_path)
        return fastas_to_seqs(fasta_file_path, stop_token=False)

    def get_pdb_file_path(self, seq_hash, predictor_structure_name='AF'):
        return get_pdb_file_path(seq_hash, predictor_structure_name=predictor_structure_name, loch_path=self.loch_path)

    def get_fasta_file_path(self, seq_hash):
        return get_fasta_file_path(seq_hash, loch_path=self.loch_path)

    def get_destress_file_path(self, seq_hash, predictor_structure_name='AF'):
        return get_destress_file_path(seq_hash, predictor_structure_name=predictor_structure_name, loch_path=self.loch_path)

    def get_md_path(self, seq_hash=None, md_param_hash=None, predictor_structure_name='AF'):
        return get_md_path(seq_hash=seq_hash, md_param_hash=md_param_hash, predictor_structure_name=predictor_structure_name, loch_path=self.loch_path)

    def run_destress(self, seq_hashes, destress_prog_dir_path, predictor_structure_name='AF'):
        for seq_hash in seq_hashes:
            destress_file_path = self.get_destress_file_path(seq_hash, predictor_structure_name=predictor_structure_name)
            if not os.path.exists(destress_file_path):
                with temp_working_directory() as tmp_dir_path:
                    src_pdb_file_path = self.get_pdb_file_path(seq_hash, predictor_structure_name=predictor_structure_name)
                    tgt_pdb_file_path = os.path.join(tmp_dir_path, f"{seq_hash}_{predictor_structure_name}.pdb")
                    shutil.copy(src_pdb_file_path, tgt_pdb_file_path)
                    
                    os.chdir(destress_prog_dir_path)
                    cmd = ['python3', 'run_destress_headless.py', '--i', tmp_dir_path]
                    result = subprocess.run(cmd, capture_output=True, check=False)
                    os.chdir(tmp_dir_path)
                    
                    if len(result.stderr) != 0:
                        print("error in run_destress")
                        import pdb
                        pdb.set_trace()
                    shutil.copy(
                        os.path.join(tmp_dir_path, 'design_data.csv'),
                        destress_file_path
                    )

    def load_pdb_id_to_seq_hash(self):
        pdb_id_to_seq_hash_file_path = os.path.join(self.loch_path, 'pdb_id_to_seq_hash.csv')
        if self.df_pdb_id_to_seq_hash is None:
            if os.path.exists(pdb_id_to_seq_hash_file_path):
                self.df_pdb_id_to_seq_hash = pd.read_csv(pdb_id_to_seq_hash_file_path).set_index('pdb_id')
            else:
                self.df_pdb_id_to_seq_hash = pd.DataFrame(columns=['pdb_id', 'seq_hash']).set_index('pdb_id')

        return pdb_id_to_seq_hash_file_path

    def set_pdb_id_to_seq_hash(self, pdb_id, seq_hash, save=True):
        pdb_id_to_seq_hash_file_path = self.load_pdb_id_to_seq_hash()
        self.df_pdb_id_to_seq_hash.loc[pdb_id] = seq_hash
        if save:
            self.df_pdb_id_to_seq_hash.to_csv(pdb_id_to_seq_hash_file_path)

    def get_seq_hash_from_pdb_id(self, pdb_id):
        self.load_pdb_id_to_seq_hash()
        if pdb_id in self.df_pdb_id_to_seq_hash.index:
            return self.df_pdb_id_to_seq_hash.loc[pdb_id]['seq_hash']
        return None

    def cp_pdb_to_dir(self, seq_hash, tgt_dir, predictor_structure_name='AF'):
        return cp_pdb_to_dir(seq_hash, tgt_dir, loch_path=self.loch_path, predictor_structure_name=predictor_structure_name)
