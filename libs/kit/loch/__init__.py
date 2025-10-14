import os
import shutil
from tqdm.auto import tqdm

from kit.data import str_to_file
from kit.log import log_info
from kit.path import join
from kit.bioinf.pdb import view_structure as view_structure_pdb
from kit.loch.path import get_fasta_file_path, get_pdb_file_path


predictors_structure = ["AF", "ESM"]
predictors_go_function = ["TransFun"]


def file_to_seq_hashes(file_path, encoding="utf-8"):
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} does not exist")

    with open(file_path, "r", encoding=encoding) as f:
        seq_hashes = f.read()

    return [seq_hash for seq_hash in seq_hashes.split("\n") if seq_hash != ""]


def seq_hashes_to_file(seq_hashes, file_path, append=False, encoding="utf-8"):
    text = "".join([f"{seq_hash}\n" for seq_hash in seq_hashes])
    str_to_file(text, file_path, append, encoding=encoding)


def copy_info(
    seq_hash_file_path,
    source_dir_path,
    target_dir_path,
    cp_sequence=True,
    cp_structure=True,
    cp_dynamics=True,
    cp_function=True,
    source_is_loch_path=True,
    target_is_loch_path=True,
):
    seq_hashes = file_to_seq_hashes(seq_hash_file_path)

    infos = ["sequences", "structures", "dynamics", "function"]

    src_paths = {
        info: (
            os.path.join(source_dir_path, info)
            if source_is_loch_path
            else source_dir_path
        )
        for info in infos
    }
    tgt_paths = {
        info: (
            os.path.join(target_dir_path, info)
            if target_is_loch_path
            else target_dir_path
        )
        for info in infos
    }

    c_fastas, c_pdb, c_dynamics, c_function = 0, 0, 0, 0
    for seq_hash in tqdm(seq_hashes):
        if cp_sequence:
            source_file_path = os.path.join(src_paths["sequences"], f"{seq_hash}.fasta")
            if os.path.exists(source_file_path):
                shutil.copyfile(
                    source_file_path,
                    join(tgt_paths["sequences"], f"{seq_hash}.fasta"),
                )
                c_fastas += 1

        if cp_structure:
            for predictor_structure in predictors_structure:
                source_file_path = os.path.join(
                    src_paths["structures"],
                    predictor_structure,
                    "pdb",
                    f"{seq_hash}_{predictor_structure}.pdb",
                )
                if os.path.exists(source_file_path):
                    shutil.copyfile(
                        source_file_path,
                        join(
                            tgt_paths["structures"],
                            predictor_structure,
                            "pdb",
                            f"{seq_hash}_{predictor_structure}.pdb",
                        ),
                    )
                    c_pdb += 1

                source_file_path = os.path.join(
                    src_paths["structures"],
                    predictor_structure,
                    "pkl",
                    f"{seq_hash}.pkl",
                )
                if os.path.exists(source_file_path):
                    shutil.copyfile(
                        source_file_path,
                        join(
                            tgt_paths["structures"],
                            predictor_structure,
                            "pkl",
                            f"{seq_hash}.pkl",
                        ),
                    )

        if cp_dynamics:
            for predictor_structure in predictors_structure:
                source_dir_path = os.path.join(
                    src_paths["dynamics"], f"{seq_hash}_{predictor_structure}"
                )
                if os.path.exists(source_dir_path):
                    shutil.copytree(
                        source_dir_path,
                        join(
                            tgt_paths["dynamics"],
                            f"{seq_hash}_{predictor_structure}",
                        ),
                    )
                    c_dynamics += 1

        if cp_function:
            for predictor_structure in predictors_structure:
                for predictor_go_function in predictors_go_function:
                    source_file_path = os.path.join(
                        src_paths["function"],
                        "GO",
                        predictor_go_function,
                        f"{seq_hash}_{predictor_structure}.txt",
                    )
                    if os.path.exists(source_file_path):
                        shutil.copyfile(
                            source_file_path,
                            join(
                                tgt_paths["function"],
                                "GO",
                                predictor_go_function,
                                f"{seq_hash}_{predictor_structure}.txt",
                            ),
                        )
                        c_function += 1

    log_info(
        f"Copied {c_fastas} fastas, {c_pdb} pdbs, {c_dynamics} dynamics, {c_function} functions"
    )


def view_structure(seq_hash, loch_path=None, predictor_structure_name="AF", output_file_path=None, width=800, height=800):
    pdb_file_path = get_pdb_file_path(seq_hash, loch_path, predictor_structure_name)
    if os.path.exists(pdb_file_path):
        view_structure_pdb(pdb_file_path, output_file_path=output_file_path, width=width, height=height)
    else:
        print(f"Could not find PDB file for {seq_hash} at {pdb_file_path}")
