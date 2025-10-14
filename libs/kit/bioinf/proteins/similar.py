import os
import pickle

from kit.path import join
from kit.bioinf.utils import chains_to_seq, get_seq_hash
from kit.bioinf.pdb import download_structure, get_similar_structures, pdb_to_seqs
from kit.bioinf.alignment.structure.tm_align import align_structures as tm_align_structures

from kit.hashes import dict_to_hash


def get_similar_proteins(
        pdb_id, pdb_dir_path,
        similar_max_score = 0.99,
        similar_min_score = 0.1,
        similar_min_tm_score = 0.9,
        similar_max_count = 10,
        max_length_diff = 0.1,
        similar_proteins_dir_path = None,
        server = r'https://files.rcsb.org'
    ):

    if similar_proteins_dir_path is not None:
        search_hash = dict_to_hash({
            'similar_max_score': similar_max_score,
            'similar_min_score': similar_min_score,
            'similar_min_tm_score': similar_min_tm_score,
            'similar_max_count': similar_max_count,
            'max_length_diff': max_length_diff
        })[:5]

        similar_proteins_file_path = join(
            similar_proteins_dir_path, 
            search_hash,
            f"{pdb_id}.pickle"
        )

        if os.path.exists(similar_proteins_file_path):
            with open(similar_proteins_file_path, 'rb') as f:
                similar_proteins = pickle.load(f)
            return similar_proteins
    
    similar_structures = {}

    pdb_file_path = download_structure(pdb_id, pdb_dir_path, server=server)
    
    seq = chains_to_seq(pdb_to_seqs(pdb_file_path)[0])
    all_similar_structures = get_similar_structures(pdb_id)
    for sim_pdb_id, s in all_similar_structures.items():
        if similar_min_score <= s <= similar_max_score:
            try:
                sim_pdb_file_path = download_structure(
                    sim_pdb_id, pdb_dir_path, 
                    file_format='pdb', 
                    source='pdb', 
                    server=server
                )
                sim_seq = chains_to_seq(pdb_to_seqs(sim_pdb_file_path)[0])
    
                if os.path.exists(sim_pdb_file_path):
                    tm_score, aligned_length, rmsd, identical = tm_align_structures(pdb_file_path, sim_pdb_file_path)    
                    min_len = len(seq) * (1-max_length_diff)
                    max_len = len(seq) * (1+max_length_diff)
                    if tm_score >= similar_min_tm_score and min_len <= len(sim_seq) <= max_len and sim_seq != seq:
                        similar_structures[get_seq_hash(sim_seq)] = (sim_pdb_id, tm_score, sim_seq)
            except Exception:
                print(f"Error in {sim_pdb_id}")
        if len(similar_structures) >= similar_max_count:
            break
    
    if similar_proteins_dir_path is not None:
        with open(similar_proteins_file_path, 'wb') as f:
            pickle.dump(similar_structures, f)

    return similar_structures