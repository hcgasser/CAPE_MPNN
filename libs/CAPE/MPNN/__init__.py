import os
import shutil
import subprocess
import tempfile

from kit.bioinf.proteins import ProteinType
from CAPE.MPNN.model import CapeMPNN


def run_parse_multiple_chains(repo_path, tmp_dir_path):
    parsed_file_path = os.path.join(tmp_dir_path, 'parsed.jsonl')
    cmd = [
        'python', os.path.join(repo_path, 'helper_scripts', 'parse_multiple_chains.py'),
        '--input_path', tmp_dir_path,
        '--output_path', parsed_file_path
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if len(result.stderr) != 0:
        raise Exception(f"error in run_parse_multiple_chains")
    return parsed_file_path


def run_assign_fixed_chains(repo_path, tmp_dir_path, designed_positions, parsed_file_path):
    chain_id_path = os.path.join(tmp_dir_path, 'chains.jsonl')
    chain_list = " ".join(designed_positions.keys())
    cmd = [
        'python', os.path.join(repo_path, 'helper_scripts', 'assign_fixed_chains.py'),
        '--input_path', parsed_file_path,
        '--output_path', chain_id_path,
        '--chain_list', chain_list
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if len(result.stderr) != 0:
        raise Exception(f"error in run_assign_fixed_chains")
    return chain_id_path


def run_make_fixed_positions_dict(repo_path, tmp_dir_path, designed_positions, parsed_file_path):
    fixed_pos_path = os.path.join(tmp_dir_path, 'fixed_pos.jsonl')

    chain_list = []
    positions_list = []
    for chain, positions in designed_positions.items():
        chain_list.append(chain)
        positions_list.append(" ".join([str(p) for p in positions]))

    chain_list = " ".join(chain_list)
    positions_list = ", ".join(positions_list)

    cmd = [
        'python', os.path.join(repo_path, 'helper_scripts', 'make_fixed_positions_dict.py'),
        '--input_path', parsed_file_path,
        '--output_path', fixed_pos_path,
        '--chain_list', chain_list,
        '--position_list', positions_list,
        '--specify_non_fixed'
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if len(result.stderr) != 0:
        raise Exception(f"error in run_make_fixed_positions_dict")
    return fixed_pos_path


def run_make_tied_positions_dict(repo_path, tmp_dir_path, parsed_file_path):
    tied_pdbs_path = os.path.join(tmp_dir_path, 'tied_pdbs.jsonl')
    cmd = [
        'python', os.path.join(repo_path, 'helper_scripts', 'make_tied_positions_dict.py'),
        '--input_path', parsed_file_path,
        '--output_path', tied_pdbs_path,
        '--homooligomer', str(1)
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if len(result.stderr) != 0:
        raise Exception(f"error in run_make_tied_positions_dict")
    return tied_pdbs_path


def get_ckpt_info(ckpt_id):
    if ':' in ckpt_id: # CAPE model
        model_id, ckpt = ckpt_id.split(':')
        return os.path.join(os.environ['PF'], 'artefacts', 'CAPE-MPNN', 'models', model_id, 'ckpts'), ckpt
    else:  # base model
        return CapeMPNN.base_model_pt_dir_path, ckpt_id


def run_mpnn(ckpt_id, pdb_input_file_path, fasta_output_file_path, seed, protein_type=None,
             designed_positions=None, sampling_temp=0.1):
    repo_path = os.path.join(os.environ['PF'], 'external', 'repos', 'ProteinMPNN')

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # copy the pdb to the temporary folder
        shutil.copy(pdb_input_file_path, os.path.join(tmp_dir_path, 'tmp.pdb'))

        # run the ProteinMPNN scripts
        parsed_file_path, tied_pdbs_path, chain_id_path, fixed_pos_path = None, None, None, None
        parsed_file_path = run_parse_multiple_chains(repo_path, tmp_dir_path)
        if designed_positions is not None:
            chain_id_path = run_assign_fixed_chains(repo_path, tmp_dir_path, designed_positions, parsed_file_path)
            fixed_pos_path = run_make_fixed_positions_dict(repo_path, tmp_dir_path, designed_positions, parsed_file_path)
        if protein_type == ProteinType.HOMOOLIGOMER:
            tied_pdbs_path = run_make_tied_positions_dict(repo_path, tmp_dir_path, parsed_file_path)

        # run the actual Protein MPNN script
        ckpt_dir_path, ckpt = get_ckpt_info(ckpt_id)
        cmd = [
            'python', os.path.join(repo_path, 'protein_mpnn_run.py'),
            '--jsonl_path', parsed_file_path,
            '--out_folder', tmp_dir_path,
            '--num_seq_per_target', str(1),
            '--sampling_temp', str(sampling_temp),
            '--seed', str(seed),  # 36 + trial
            '--batch_size', str(1),
            '--path_to_model_weights', ckpt_dir_path,
            '--model_name', ckpt
        ]
        if tied_pdbs_path is not None:
            cmd += ['--tied_positions_jsonl', tied_pdbs_path]
        if chain_id_path is not None:
            cmd += ['--chain_id_jsonl', chain_id_path]
        if fixed_pos_path is not None:
            cmd += ['--fixed_positions_jsonl', fixed_pos_path]

        result = subprocess.run(cmd, capture_output=True, check=False)
        if len(result.stderr) != 0:
            raise Exception(f"error in protein_mpnn_run: {ckpt_id} {os.path.basename(pdb_input_file_path)}")

        # copy the results to the destination
        seqs_file_path = os.path.join(tmp_dir_path, 'seqs', 'tmp.fa')
        if os.path.exists(seqs_file_path):
            shutil.copy(seqs_file_path, fasta_output_file_path)
        else:
            raise Exception(f"error: {ckpt_id} {os.path.basename(pdb_input_file_path)}")
