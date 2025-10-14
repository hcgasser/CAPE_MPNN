import os
from tqdm.auto import tqdm

from kit.path import join
from kit.bioinf.fasta import fasta_to_df
from kit.bioinf.proteins import Protein

from CAPE.Eval.MPNN import E
from CAPE.MPNN import run_mpnn
from CAPE.Eval.MPNN.design import Design


def get_designed_positions(seq, predictor_setup, immuno_setup,
                           design_epitope_window=None, design_anchor_window=None):
    if design_epitope_window is None and design_anchor_window is None:
        return None, None

    assert design_anchor_window is None or design_epitope_window is None

    chains = seq.split('/')

    designed_positions, visible_peptides = {}, {}
    for chain, seq in enumerate(chains):
        chain = chr(chain + ord('A'))

        vp = predictor_setup['mhc_1'].seq_presented(seq, immuno_setup['mhc_1'])
        max_pos = len(seq)
        dp = set()

        for visible_peptide, mhc_allele, rand, pos in vp:
            first_idx = pos - len(visible_peptide) + 1
            term_idx = pos

            if design_epitope_window is not None:
                # the +1 here are because the first amino acid is indexed as 1 in ProteinMPNN
                # this is different from our indexing
                to_design_start = max(first_idx - design_epitope_window + 1, 1)
                to_design_stop = min(term_idx + design_epitope_window + 1, max_pos)
                to_design = list(range(to_design_start, to_design_stop + 1))  # the +1 here is necessary as range is excluding end

            if design_anchor_window is not None:
                second_idx = first_idx + 1
                b_pocket = [max(x + 1, 1) for x in range(second_idx - design_anchor_window, second_idx + design_anchor_window + 1)]
                f_pocket = [min(x + 1, max_pos) for x in range(term_idx - design_anchor_window, term_idx + design_anchor_window + 1)]
                to_design = b_pocket + f_pocket

            dp = dp.union(set(to_design))

        designed_positions[chain] = sorted(list(dp))
        visible_peptides[chain] = vp

    return designed_positions, visible_peptides


def call_mpnn(pdb_ids, ckpt_ids, trials, protein_infos, fasta_output_path,
              predictor_setup=None, immuno_setup=None, overwrite=False,
              design_epitope_window=None, design_anchor_window=None):
    assert design_anchor_window is None or design_epitope_window is None

    designed_positions_name = 'all'
    if design_epitope_window is not None:
        designed_positions_name = f'epitope +/- {design_epitope_window}'
    if design_anchor_window is not None:
        designed_positions_name = f'anchors +/- {design_anchor_window}'
    subfolder_name = designed_positions_name.replace(" +/- ", "_pm_")

    for pdb_id in pdb_ids:
        print(pdb_id)
        design_data = Design.designs['data'][pdb_id]
        seq_data = list(Design.designs['data'][pdb_id].seqs.values())[0]

        protein_info = protein_infos[pdb_id]

        seq_hash_data = design_data.get_seq_hash_data()
        pdb_input_file_path = Protein.loch.get_pdb_file_path(seq_hash_data, predictor_structure_name='exp')
        #pdb_input_file_path = os.path.join(pdb_dir_path, f"{pdb_id}.pdb")

        designed_positions, _ = get_designed_positions(
            seq_data,
            predictor_setup,
            immuno_setup,
            design_epitope_window=design_epitope_window,
            design_anchor_window=design_anchor_window
        )

        if not os.path.exists(pdb_input_file_path):
            print(f"{pdb_id} pdb does not exist")
            break
        for ckpt_id in tqdm(ckpt_ids):
            design = Design.get(ckpt_id, design_data)
            for trial in trials:
                seed = 36 + trial
                fasta_output_file_path = join(fasta_output_path, subfolder_name, ckpt_id, f"{pdb_id}_{seed}.fasta")
                if overwrite or not os.path.exists(fasta_output_file_path):
                    run_mpnn(
                        ckpt_id,
                        pdb_input_file_path,
                        fasta_output_file_path,
                        seed,
                        protein_type=protein_info[1],
                        designed_positions=designed_positions)

                if not os.path.exists(fasta_output_file_path):
                    print(f"{ckpt_id:30s} {pdb_id} {trial} does not exist!")
                else:
                    seq = fasta_to_df(fasta_output_file_path).iloc[1].seq

                    assert len(seq) == len(seq_data)

                    # add to memory
                    design.add_trial(seq, designed_positions_name)

                    # add to DB
                    seq_hash = E.DB.add_seq(seq)
                    E.DB.add_seq_to_list('specific', ckpt_id, pdb_id, designed_positions_name, trial, seq_hash)

                    # add to loch
                    E.LOCH.add_entry(seq=seq)