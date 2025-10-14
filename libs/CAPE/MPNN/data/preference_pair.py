from collections import defaultdict

import numpy as np

import torch
import torch.nn.functional as F

from CAPE.MPNN.data.aux import S_to_seqs


def get_seqs_from_sample_dicts(sample_dicts, chain_encoding_all):
    seqs = []
    for sample_dict in sample_dicts:
        seqs.append(
            S_to_seqs(
                sample_dict['S'].unsqueeze(0),
                chain_encoding_all.unsqueeze(0)
            )[0]
        )

    return seqs


class PreferencePair:
    instances_queue = []
    def __init__(self, 
                 structure_batch_list, 
                 sample_dicts, 
                 base_log_probs,
                 immuno,
                 proteome_tree=None):
        self.length = structure_batch_list[3]
        self.structure_batch_list = self.set_structure_batch_list(structure_batch_list)
        
        self.sample_dicts = [
            {
                'S': sample_dict['S'][:self.length],
                'log_probs': sample_dict['log_probs'][:self.length]
            } for sample_dict in sample_dicts
        ]
        self.seqs = get_seqs_from_sample_dicts(sample_dicts, structure_batch_list[5])
        self.base_log_probs = [_base_log_probs[:self.length] for _base_log_probs in base_log_probs]
        self.preferences = []

        self.idx_desc = []  # position 0 will be the preferred example

        self.immuno = immuno
        self.proteome_tree = proteome_tree

        self.queue_preference_prediction()

    @staticmethod
    def calc_preference_predictions():
        predictors_mhc_1 = set()
        for instance in PreferencePair.instances_queue:
            predictors_mhc_1.add(instance.immuno['MHC_I'][0])
            instance.preferences = [
                get_preference_of_seq(seq, instance.immuno, instance.proteome_tree) 
                for seq in instance.seqs
            ]
            if len(set(instance.preferences)) > 1: # prefer the first example if both have same preference
                instance.idx_desc = np.argsort(instance.preferences)[::-1]
            else:
                instance.idx_desc = np.array([0, 1])

        PreferencePair.instances_queue = []
        # it is necessary to reset percentiles to prevent running out of memory
        for predictor_mhc_1 in predictors_mhc_1:
            predictor_mhc_1.percentiles = defaultdict(lambda: {})

    def queue_preference_prediction(self):
        PreferencePair.instances_queue.append(self)
        for seq in self.seqs:
            self.immuno['MHC_I'][0].queue_seq(seq, self.immuno['MHC_I'][1], self.immuno['MHC_I'][2])

    def set_structure_batch_list(self, structure_batch_list):
        self.structure_batch_list = []
        for element in structure_batch_list:
            if isinstance(element, torch.Tensor) and element.dim() > 0:
                element = element[:self.length]
            self.structure_batch_list.append(element)
        return self.structure_batch_list

    def get_structure_batch_list(self, length=None, device='cpu'):
        result = []
        if length is None:
            result = self.structure_batch_list
        else:
            if length < self.length:
                raise Exception("length cannot be shorter than example length")
            else:
                pad = [0, length - self.length]
                for element in self.structure_batch_list:
                    # if the element is a tensor and the tensor has a dimension representing each element
                    if isinstance(element, torch.Tensor):
                        if element.dim() > 0 and element.shape[0] == self.length:
                            element = F.pad(element, [0, 0] * (element.dim() - 1) + pad, value=0)
                        element = element.to(device)
                    result.append(element)
        return result

    def get_preference_w(self):
        return self.preferences[self.idx_desc[0]]
    
    def get_preference_l(self):
        return self.preferences[self.idx_desc[1]]

    def get_sample_dict_w(self, length=None, device='cpu'):
        return get_sample_dict(self, self.sample_dicts[self.idx_desc[0]], length, device)

    def get_sample_dict_l(self, length=None, device='cpu'):
        return get_sample_dict(self, self.sample_dicts[self.idx_desc[1]], length, device)

    def get_base_log_probs_w(self, length=None, device='cpu'):
        return get_log_probs(self, self.base_log_probs[self.idx_desc[0]], length, device)

    def get_base_log_probs_l(self, length=None, device='cpu'):
        return get_log_probs(self, self.base_log_probs[self.idx_desc[1]], length, device)
    
def get_sample_dict(example, sample_dict, length=None, device='cpu'):
    result = {}
    if length is None:
        result = sample_dict
    else:
        if length < example.length:
            raise Exception("length cannot be shorter than example length")
        pad = [0, length - example.length]
        for key, element in sample_dict.items():
            # if the element is a tensor and the tensor has a dimension representing each element
            if isinstance(element, torch.Tensor):
                if element.dim() > 0 and element.shape[0] == example.length:
                    element = F.pad(element, [0, 0] * (element.dim() - 1) + pad, value=0)
                element = element.to(device)
            result[key] = element
    return result

def get_log_probs(example, log_probs, length=None, device='cpu'):
    if length is not None:
        if length < example.length:
            raise Exception("length cannot be shorter than example length")
        pad = [0, length - example.length]
        if log_probs.dim() > 0 and log_probs.shape[0] == example.length:
            log_probs = F.pad(log_probs, [0, 0] * (log_probs.dim() - 1) + pad, value=0)
    return log_probs.to(device)

def get_preference_of_seq(seq, immuno, proteome_tree=None):
    predictor_mhc_1, alleles_mhc_1, peptide_lengths = immuno['MHC_I']
    visibility = predictor_mhc_1.seq_presented(seq, alleles_mhc_1, lengths=peptide_lengths)
    if proteome_tree is not None:
        visibility = [v for v in visibility if not proteome_tree.has_kmer(v[0])]
    return -len(visibility)



