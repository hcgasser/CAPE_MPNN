import os

from kit.bioinf.fasta import seqs_to_fasta
from kit.loch.utils import get_seq_hash
from kit.loch.path import get_fasta_file_path, get_pdb_file_path, get_destress_file_path, get_function_path, get_md_path


def add_seq(seq, loch_path=None):
    seq_hash = get_seq_hash(seq)
    fasta_file_path = get_fasta_file_path(seq_hash, loch_path=loch_path)
    seqs_to_fasta(seq, fasta_file_path)
    return seq_hash

def rm_seq(seq_hash, loch_path=None):
    fasta_file_path = get_fasta_file_path(seq_hash, loch_path=loch_path)
    if os.path.exists(fasta_file_path):
        os.remove(fasta_file_path)

    for predictor_structure_name in PREDICTOR_STRUCTURE_NAMES:
        pdb_file_path = get_pdb_file_path(seq_hash, loch_path=loch_path, predictor_structure_name=predictor_structure_name)
        if os.path.exists(pdb_file_path):
            os.remove(pdb_file_path)

        destress_file_path = get_destress_file_path(seq_hash, loch_path=loch_path, predictor_structure_name=predictor_structure_name)
        if os.path.exists(destress_file_path):
            os.remove(destress_file_path)

    function_path = get_function_path(seq_hash, loch_path=loch_path)
    if os.path.exists(function_path):
        os.remove(function_path)

    md_path = get_md_path(seq_hash, loch_path=loch_path)
    if os.path.exists(md_path):
        shutil.rmtree(md_path)
