""" 
Functions to generat sequences for the general dataset
"""

from tqdm.auto import tqdm

import kit
from kit.nn import move_dict_to_device, move_list_to_device

from CAPE.Eval.MPNN import E
from CAPE.MPNN.ProteinMPNN.protein_mpnn_utils import _S_to_seq


def generate_general(batches, split, ckpts, ckpt_ids, max_batches=None, add_seqs=True):
    """
    Generate sequences for the general dataset

    Args:
        batches (dict): dictionary of general sequences data batches (split is the key)
        split (Split): split of the data to generate the sequences for
        ckpts (dict): dictionary that holds the checkpoints
        ckpt_ids (list): checkpoint ids of the checkpoints to generate the sequences with
        max_batches (int): maximum number of batches to generate
        add_seqs (bool): add sequences to the database
    """


    _done = {}
    for ckpt_id in ckpt_ids:
        _done[ckpt_id] = list(E.DB.get_list(f"general:{split}", ckpt_id, 'all').id)

    nr = 1
    candidates_mono = []
    candidates_oligo = []

    c_proteins = 0
    for j, d_batch in tqdm(enumerate(batches[split]), leave=False):
        batch = list(d_batch['tied_featurize'])
        move_list_to_device(batch, kit.DEVICE)
        X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask, \
            temperature, omit_AAs_np, bias_AAs_np, chain_M_pos, \
            omit_AA_mask, bias_by_res_all = batch

        batch_size = S.shape[0]

        # forward pass batch
        sample_dicts = {}
        for ckpt_id in tqdm(ckpt_ids):
            model = ckpts[ckpt_id]
            # is there any example that has to be sampled in this batch?
            if (add_seqs and 
             any(_nr not in _done[ckpt_id] for _nr in range(nr, nr + batch_size))):
                model.to(kit.DEVICE)
                sample_dict = model.sample(
                    X, randn_2, S, chain_M,
                    chain_encoding_all, residue_idx,
                    mask=mask,
                    temperature=temperature,
                    omit_AAs_np=omit_AAs_np,
                    bias_AAs_np=bias_AAs_np,
                    chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_AA_mask,
                    bias_by_res=bias_by_res_all
                )
                move_dict_to_device(sample_dict, "cpu")
                model.to('cpu')
            else:
                sample_dict = None
            sample_dicts[ckpt_id] = sample_dict

        # add sequences to database and loch
        for b in range(batch_size):

            # data sequences
            data_seq = _S_to_seq(S[b], mask[b])
            data_seq.replace("-", "X")

            # add sequence to DB
            data_seq_hash = E.DB.add_seq(data_seq)
            E.DB.add_seq_to_list(
                f"general:{split}", "data", nr, 'all', 1, data_seq_hash
            )  # add sequence to list

            # add to Loch
            E.LOCH.add_entry(seq=data_seq)  # add sequence to loch

            # could this be a monomer?
            if chain_encoding_all[b].max() == 1:
                name = d_batch['names'][b]
                if name[-1] == 'A':
                    candidates_mono.append((name, len(data_seq)))
            else:
                name = d_batch['names'][b]
                candidates_oligo.append((name, len(data_seq)))

            # add checkpoint generated sequences
            for ckpt_id in ckpt_ids:
                sample_dict = sample_dicts[ckpt_id]

                if nr not in _done[ckpt_id]:
                    seq = _S_to_seq(sample_dict['S'][b], mask[b])

                    assert len(seq) == len(data_seq)

                    # add to DB
                    seq_hash = E.DB.add_seq(seq)  # add sequence to DB
                    E.DB.add_seq_to_list(
                        f"general:{split}", ckpt_id, nr, 'all', 1, seq_hash
                    )  # add sequence to list

                    # add to Loch
                    E.LOCH.add_entry(seq=seq)  # add sequence to loch

            c_proteins += b
            nr += 1

        if max_batches is not None and (j + 1 >= max_batches):
            break

    print(f"{c_proteins} proteins in set")

    return candidates_mono, candidates_oligo