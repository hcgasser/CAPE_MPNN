import torch
import numpy as np


class PreferencesDataSet:
    def __init__(self, examples, batch_size=100, device='cpu'):
        self.examples = examples
        self.size = len(examples)
        self.lengths = [example.length for example in examples]
        self.batch_size = batch_size
        self.device = device

        # create clusters
        clusters, batch, batch_max_size = [], [], 0
        for ix in np.random.permutation(range(len(examples))):
            size = self.lengths[ix]
            if max(size, batch_max_size) * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max_size = max(size, batch_max_size)
            else:
                clusters.append(batch)
                if size <= self.batch_size:
                    batch, batch_max_size = [ix], size
                else:
                    batch, batch_max_size = [], 0
                    
        if len(batch) > 0:
            clusters.append(batch)

        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)
    
    def __getitem__(self, j):
        return self.clusters[j]

    def collate(self, cluster):
        cluster = [e for c in cluster for e in c]
        return PreferencesDataSet.get_batch_from_preference_pairs([self.examples[i] for i in cluster], self.device)
    
    @staticmethod
    def get_batch_from_preference_pairs(preference_pairs, device):
        max_length = np.max([preference_pair.length for preference_pair in preference_pairs])

        structure_batch_lists = []
        sample_dicts_w, sample_dicts_l = [], []
        base_log_probs_w, base_log_probs_l = [], []
        for preference_pair in preference_pairs:
            structure_batch_lists.append(preference_pair.get_structure_batch_list(max_length, device))
            sample_dicts_w.append(preference_pair.get_sample_dict_w(max_length, device))
            sample_dicts_l.append(preference_pair.get_sample_dict_l(max_length, device))
            base_log_probs_w.append(preference_pair.get_base_log_probs_w(max_length, device))
            base_log_probs_l.append(preference_pair.get_base_log_probs_l(max_length, device))

        structure_batch_list = []
        for i, element in enumerate(structure_batch_lists[0]):
            if isinstance(element, torch.Tensor):
                element = torch.stack([bl[i] for bl in structure_batch_lists], dim=0)
            if isinstance(element, list):
                element = [bl[i] for bl in structure_batch_lists]

            structure_batch_list.append(element)

        sample_dict_w = {
            'S': torch.stack([sd['S'] for sd in sample_dicts_w], dim=0),
            'log_probs': torch.stack([sd['log_probs'] for sd in sample_dicts_w], dim=0),
        }

        sample_dict_l = {
            'S': torch.stack([sd['S'] for sd in sample_dicts_l], dim=0),
            'log_probs': torch.stack([sd['log_probs'] for sd in sample_dicts_l], dim=0),
        }

        base_log_probs_w = torch.stack(base_log_probs_w, dim=0)
        base_log_probs_l = torch.stack(base_log_probs_l, dim=0)
        
        return (
            max_length, 
            structure_batch_list, 
            sample_dict_w, 
            base_log_probs_w, 
            sample_dict_l, 
            base_log_probs_l, 
            preference_pairs
        )

    