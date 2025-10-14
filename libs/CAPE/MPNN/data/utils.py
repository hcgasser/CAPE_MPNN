# ----------------------------------------------------------------
# Portions of this file were adapted from https://github.com/dauparas/ProteinMPNN
# Copyright (c) 2022 Justas Dauparas
# Licensed under the MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------



import queue
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

import re
import numpy as np

import torch
from torch.utils.data import DataLoader

import kit
from kit.data import Split
from kit.log import log_info
from kit.nn import move_dict_to_device, move_list_to_device

from CAPE.MPNN.overwrite import tied_featurize
from CAPE.MPNN.data.preference_pair import PreferencePair
from CAPE.MPNN.data.dataset import PreferencesDataSet
from CAPE.MPNN.data.aux import S_to_seqs

from CAPE.MPNN.utils import ModelManager
from CAPE.MPNN.model import CapeMPNN
from CAPE.MPNN.ProteinMPNN.training.utils import worker_init_fn, get_pdbs, loader_pdb, \
    build_training_clusters, PDB_dataset, StructureDataset


EXECUTOR = None
QUEUES = {}

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
OMIT_AAs_NP = np.array([AA in "X" for AA in ALPHABET]).astype(np.float32)
BIAS_AAs_NP = np.zeros(len(ALPHABET))


def get_data_loaders_pdb(data_path, splits, rescut, debug):
    """Builds data loaders for the PDB dataset.
    
    :param data_path: path to the PDB dataset
    :param splits: list of splits to build data loaders for (e.g. [Split.TRAIN, Split.VAL])
    :param rescut: resolution cutoff for PDBs
    :param debug: whether to use the debug mode
    :return: data loaders for the PDB dataset
    """
    params = {
        "LIST": f"{data_path}/list.csv",
        "VAL": f"{data_path}/valid_clusters.txt",
        "TEST": f"{data_path}/test_clusters.txt",
        "DIR": f"{data_path}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": rescut,  # resolution cutoff for PDBs
        "HOMO": 0.70  # min seq.id. to detect homo chains
    }

    DATA_LOADER_PARAM = {'batch_size': 1,
                        'shuffle': True,
                        'pin_memory': False,
                        'num_workers': 4}

    # train, valid, test are dictionaries
    # their keys are the clusters (specified in valid_clusters.txt, test_clusters.txt, else train)
    # the values are lists of [CHAINID, HASH]
    # CHAINID is actually PDBID_CHAINID and HASH a 'unique 6-digit seq hash' (see data README)
    clusters_train, clusters_valid, clusters_test = build_training_clusters(params, debug)

    data_loaders = {}

    # The datasets specified below are very light-weight
    # Their length is the number of clusters 
    # When an item is requested, take this position's cluster and sample a random chain from it
    # The WHOLE PROTEIN to which the chain belongs is returned
    # For this, loader_pdb is called (e.g. loader_pdb(['5naf_B', '12123'], params)  )
    #   - returns {'seq': ..., 'xyz': ..., 'idx': ..., 'masked': ..., 'label': ...}
    #       - seq: str
    #         is the sequence of the protein (e.g. 'MANVYDWFQERLEIQALADD...GPL')
    #       - xyz: Tensor, shape (length of sequence, 14, 3)... 14 is for the heavy atoms?
    #         atoms' 3D coordinates. Unknown/unused atoms (e.g. in GLY only 4 are need) are nan
    #       - idx: Tensor, shape (length of sequence)
    #         specifies the chain each amino acid in the sequence belongs to
    #         CAUTION: chain A is not necessarily 0... they get mixed up
    #       - masked: Tensor 1 dimension, holds the chain numbers that are considered homologous
    #       - label: is simple the PDBID_CHAINID (e.g. '5naf_B')
    #   - goes into the 'data_path' directory and opens the metadata-file of the PDBID
    #   - there can be multiple assemblies for a PDBID
    #       - if none of the assemblies have a chain with the chains name, only the chain is returned
    #       - if multiple assemblies have a chain with the chains name, a random assembly is used
    #         and its chains concatenated (randomly)
    #       - the assembly can also specify transformations (translation and rotation) 
    #         to apply to chains in the assembly. All variants are concatenated
    #       - based on the sequence identity calculated with TMalign, homology between chains is judged
    #           - if the seq identity is above the homology theshold, the chain number is added to the 'masked'
    if Split.TRAIN in splits:
        train_set = PDB_dataset(list(clusters_train.keys()), loader_pdb, clusters_train, params)
        data_loaders[Split.TRAIN] = torch.utils.data.DataLoader(
            train_set, worker_init_fn=worker_init_fn, **DATA_LOADER_PARAM
        )
    if Split.VAL in splits:
        valid_set = PDB_dataset(list(clusters_valid.keys()), loader_pdb, clusters_valid, params)
        data_loaders[Split.VAL] = torch.utils.data.DataLoader(
            valid_set, worker_init_fn=worker_init_fn, **DATA_LOADER_PARAM
        )
    if Split.TEST in splits:
        test_set = PDB_dataset(list(clusters_test.keys()), loader_pdb, clusters_test, params)
        data_loaders[Split.TEST] = torch.utils.data.DataLoader(
            test_set, worker_init_fn=worker_init_fn, **DATA_LOADER_PARAM
        )
    return data_loaders


def get_structure_data_sets(data_loaders_pdbs, max_protein_length, num_examples_per_epoch, 
        multithreading):
    global EXECUTOR, QUEUES

    start_threads = 1

    pdb_dicts = {}
    splits = list(data_loaders_pdbs.keys())

    if multithreading:
        if EXECUTOR is None:
            EXECUTOR = ProcessPoolExecutor(max_workers=12)

            start_threads = 3
            for split in splits:
                QUEUES[split] = queue.Queue(maxsize=start_threads)

        for _ in range(start_threads):
            for split in splits:
                QUEUES[split].put_nowait(
                    EXECUTOR.submit(get_pdbs, data_loaders_pdbs[split], 1, max_protein_length, num_examples_per_epoch))

        pdb_dicts = {split: QUEUES[split].get().result() for split in splits}
    else:
        pdb_dicts = {split: get_pdbs(
                data_loaders_pdbs[split], 1, max_protein_length, num_examples_per_epoch
            ) for split in splits
        }

    structure_data_sets = {split: StructureDataset(
        pdb_dicts[split], truncate=None, max_length=max_protein_length) for split in splits}

    return structure_data_sets


def get_data_loader_preferences(preference_pairs, batch_size):
    preference_data_set = PreferencesDataSet(preference_pairs, batch_size=batch_size, device=kit.DEVICE)
    collate_fn = preference_data_set.collate
    preference_data_loader = DataLoader(preference_data_set, collate_fn=collate_fn)
    return preference_data_loader


def get_avg_preference_score(preference_pairs):
    preference_scores = []
    for preference_pair in preference_pairs:
        preference_scores += [preference_pair.get_preference_w(), preference_pair.get_preference_l()]
    avg_preference_score = np.mean(preference_scores)
    return avg_preference_score


def get_preference_pairs(model, immuno, split, calc_prediction_every=20, preference_sampling_method=None):
    model_cape = model.torch
    model_base = ModelManager.models[model.dpo_hparams.base_model_name].torch
    data_loader_structures = model.dpo_structure_data_loaders[split]
    temperature = model.dpo_hparams.temperature
    proteome_tree = model.proteome_trees[model.dpo_hparams.proteome_file_name] if model.dpo_hparams.proteome_file_name is not None else None

    preference_pairs = []
    if preference_sampling_method is None:
        preference_sampling_method = 'C_B_D_all+C_B_1_all'

    # Example for preference_sampling_method: "[C_B_D_epitopes+C_B_1_epitopes;10][B_B_D_epitopes+B_B_1_all;90]"
    # in 10% of cases do the first method, in 90% of cases the second one
    parts = re.findall(r"\[([A-Za-z0-9\_\+]+);(\d+)\]*", preference_sampling_method)
    if len(parts) == 0:
        parts = [(preference_sampling_method, 100)]

    methods, probs = [], []
    for part in parts:
        method, prob = part
        probs.append(int(prob)/100.)

        instructions = []
        for instruction in method.split("+"):
            # sampling model, probability model, sequence origin, designed positions
            h = re.findall("^([BCD])_([BC])_([D12])_(.+)$", instruction)
            assert len(h) == 1
            instructions.append(h[0])

        methods.append(instructions)

    assert np.isclose(np.sum(probs), 1.)

    for idx, batch in enumerate(data_loader_structures):
        batch_structures = list(tied_featurize(batch, kit.DEVICE, None))

        method = methods[np.random.choice(len(methods), p=probs)]
        instruction_1, instruction_2 = method

        sample_dicts, base_log_probs = {}, {}
        sampling_model, probability_model, sequence_origin, designed_positions = instruction_1
        sample_dicts['1'], base_log_probs['1'] = sample_preference_example(
            sampling_model, probability_model, sequence_origin, designed_positions,
            sample_dicts, model_cape, model_base, batch_structures, temperature, immuno
        )

        sampling_model, probability_model, sequence_origin, designed_positions = instruction_2
        sample_dicts['2'], base_log_probs['2'] = sample_preference_example(
            sampling_model, probability_model, sequence_origin, designed_positions,
            sample_dicts, model_cape, model_base, batch_structures, temperature, immuno
        )

        move_list_to_device(batch_structures, "cpu", detach=True)

        preference_pairs += get_preference_pairs_from_batch(
            batch_structures,
            sample_dicts['1'],
            base_log_probs['1'],
            sample_dicts['2'],
            base_log_probs['2'],
            immuno,
            proteome_tree=proteome_tree
        )

        if idx % calc_prediction_every == 0:
            PreferencePair.calc_preference_predictions()
            
    PreferencePair.calc_preference_predictions()

    return preference_pairs


def sample_preference_example(
        sampling_model, probability_model, sequence_origin, designed_positions,
        sample_dicts, model_cape, model_base, batch_structures, temperature, immuno):
    _designed_positions = None

    # determine the models to use
    model_sampling = {
        'B': model_base,  # ProteinMPNN
        'C': model_cape,  # CAPE-MPNN
        'D': None  # Data
    }[sampling_model]

    model_probs = model_cape if probability_model == 'C' else model_base

    # determine the template sequences
    _batch_structures = batch_structures
    if sequence_origin != 'D':  # use some other sequence than the data sequence as base
        _batch_structures[1] = sample_dicts[sequence_origin]['S'].clone().to(
            _batch_structures[1].device
        )
    S = _batch_structures[1]
    seqs_template = S_to_seqs(S, _batch_structures[5])

    # determine the designed positions
    if designed_positions != 'all':
        predictor_mhc_1, alleles_mhc_1, peptide_lengths = immuno['MHC_I']
        _designed_positions = []

        for seq in seqs_template:
            if designed_positions == 'single':
                designed_pos = [np.random.randint(len(seq.replace('/', '')))]
            else:
                pos_epitopes, pos_anchors, pos_unknown = \
                    predictor_mhc_1.get_presented_positions(
                        seq,
                        alleles_mhc_1,
                        lengths=peptide_lengths
                    )

                if designed_positions == 'epitopes':
                    designed_pos = pos_epitopes
                elif designed_positions == 'nepitopes':
                    designed_pos = sorted(set(range(len(seq.replace('/', '')))) - set(pos_epitopes))
                else:
                    raise Exception(f"Unknown designed_position: {designed_positions}")

            _designed_positions.append(designed_pos)

    # sample the example
    sample_dict, log_probs = sample_dict_and_probs(
        model_sampling,
        model_probs,
        _batch_structures,
        temperature,
        designed_positions=_designed_positions
    )

    # assert that the seqs have the right lengths and that all not designed AA stayed the same
    seqs = S_to_seqs(sample_dict["S"], _batch_structures[5])

    for i, (seq_template, seq) in enumerate(zip(seqs_template, seqs)):
        seq_lengths = _batch_structures[3].tolist()
        seq_template, seq = seq_template.replace('/', ''), seq.replace('/', '')

        # assert len(seq_template) == len(seq) == seq_lengths[i]
        if not (len(seq_template) == len(seq) == seq_lengths[i]):
            print(f"len(seq_template): {len(seq_template)}; len(seq): {len(seq)}; seq_lengths[i]: {seq_lengths[i]}")
            print(f"seq_template: {seq_template}")
            print(f"seq: {seq}")
            print(f"S: {S}")
            print(f"sample_dict['S']: {sample_dict['S']}")
            assert False

        if designed_positions != 'all':
            not_designed_positions = [p for p in range(seq_lengths[i]) if p not in _designed_positions[i]]
            assert [seq_template[p] for p in not_designed_positions] == [seq[p] for p in not_designed_positions]

    return sample_dict, log_probs


def sample_dict_and_probs(model_sample, model_log_probs, batch_structures, temperature, designed_positions=None):
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = batch_structures
    B, L = S.shape

    mask_for_sample = mask.clone()  # we only use this mask for sampling a new protein, not for calculating the base_log_probs
    if designed_positions is not None:
        mask_for_sample = torch.zeros_like(mask)
        for b in range(B):
            for p in designed_positions[b]:
                mask_for_sample[b, p] = mask[b, p]

    model_log_probs.eval()
    model_log_probs.to("cpu")

    # sample examples
    randn_2 = torch.randn(chain_M.shape, device=X.device)

    if model_sample is not None:
        model_sample.eval()
        model_sample.to(kit.DEVICE)
        sample_dict = model_sample.sample(X, randn_2, S, chain_M,
                        chain_encoding_all, residue_idx,
                        mask=mask_for_sample,
                        temperature=temperature,
                        omit_AAs_np=OMIT_AAs_NP,
                        bias_AAs_np=BIAS_AAs_NP,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        bias_by_res=bias_by_res_all)
        model_sample.to("cpu")
    else:
        probs = torch.nn.functional.one_hot(S, 21).to(torch.float)
        sample_dict = {
            "S": S,
            "probs": probs,
            "log_probs": torch.log(probs),
            "decoding_order": torch.arange(S.shape[-1])[None,:].repeat((B,1))
        }

    # get the log probabilities for these samples from the original model
    S_tmp = sample_dict['S'].detach()
    move_dict_to_device(sample_dict, "cpu", detach=True)
    decoding_order = torch.arange(S_tmp.shape[1]).repeat(B, 1).to(kit.DEVICE)

    model_log_probs.to(kit.DEVICE)
    base_log_probs = model_log_probs(
        X, S_tmp, mask, chain_M, residue_idx, chain_encoding_all, decoding_order,
        omit_AA_mask=omit_AA_mask, temperature=temperature
    )
    model_log_probs.to("cpu")

    return sample_dict, base_log_probs.detach().to('cpu')


def get_preference_pairs_from_batch(
        batch_structures, 
        sample_dict_1, base_log_probs_1, 
        sample_dict_2, base_log_probs_2, 
        immuno,
        proteome_tree=None
):
    # define which elements of the batch are 'tied' (shared)
    tied_element = [
        False, False, False, False, False, False, False, \
        False, False, False, \
        False, False, False, False, \
        False, False, False, False, \
        False, True]

    B, l_examples = len(sample_dict_1["S"]), []
    for idx in range(B):
        # try:
            example_structure = [
                    (element[idx] if not tied_element[j] else element) 
                for j, element in enumerate(batch_structures)]
            example_sample_dict_1 = {key: value[idx] for key, value in sample_dict_1.items()}
            example_sample_dict_2 = {key: value[idx] for key, value in sample_dict_2.items()}

            example_base_log_probs_1 = base_log_probs_1[idx]
            example_base_log_probs_2 = base_log_probs_2[idx]

            l_examples.append(PreferencePair(example_structure,
                [example_sample_dict_1, example_sample_dict_2],
                [example_base_log_probs_1, example_base_log_probs_2], 
                immuno, proteome_tree=proteome_tree))

        # except Exception as e:
        #     log_info(f"Exception: {e}")
        #     log_info(f"Batch size {B}")
        #     for element in batch_structures:
        #         log_info(f"  vs {len(element)}")
        #     log_info(str(sample_dict_1["S"]))
        #     log_info(sample_dict_1["S"].shape)
        #     kit.pdb.set_trace()


    return l_examples


def shutdown():
    EXECUTOR.shutdown()
