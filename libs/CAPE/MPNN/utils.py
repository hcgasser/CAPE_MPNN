import os

from kit.bioinf import AA1_FULL
from kit.bioinf.fasta import read_fasta
from kit.hashes import dict_to_hash, str_to_hash
from kit.path import join
from kit.data import Split, DD
from kit.data.trees import PrefixTree
import kit.globals as G

from CAPE.MPNN.model import CapeMPNN

def get_proteome_hash(proteome_file_name):
    return str_to_hash(proteome_file_name, truncate=5)

def load_proteome_tree(proteome_file_name, alphabet=AA1_FULL):
    proteome_file_path = os.path.join(G.ENV.INPUT, "proteomes", proteome_file_name)
    proteome_hash = get_proteome_hash(proteome_file_name)
    proteome = read_fasta(proteome_file_path, stop_token=False, evaluate=False, return_df=True)

    PrefixTree.set_alphabet(alphabet)
    proteome_tree = PrefixTree()

    wrong_seqs = []
    for seq, _ in proteome.iterrows():
        missing = proteome_tree.add_seq(seq, 10)
        if len(missing) > 0:
            wrong_seqs.append(seq)
    
    return proteome_hash, proteome_tree

class ModelManager:
    models = {}
    proteome_trees = {}

    dpo_hparams_names = None
    dpo_hparams_std = {
        'proteome_file_name': None
    }

    def __init__(self, dpo_hparams=None, id=None, load_ckpt_id=None):
        if ModelManager.dpo_hparams_names is None:
            raise Exception("ModelManager.hparams_names must be set")
        
        # bust be either
        #   a base_model
        #   a list of given DPO hyper-parameters
        #   or a loaded checkpoint
        if id is None and dpo_hparams is None and load_ckpt_id is None:
            raise Exception("Either dpo_hparams, id or load_ckpt_id must be set")
        
        if id is not None:
            self.is_base_model = True
            self.dpo_hparams = None
            self.id = id
        else:
            self.is_base_model = False

            # load a checkpoint
            if load_ckpt_id is not None:
                model_id, ckpt = load_ckpt_id.split(":")
                dpo_hparams = DD.from_yaml(os.path.join(G.ENV.ARTEFACTS, 'models', model_id, "dpo_hparams.yaml"))

            # set standard dpo-hparams
            for _dpo_hparam_name, _dpo_hparam_value in ModelManager.dpo_hparams_std.items():
                if _dpo_hparam_name not in dpo_hparams:
                    dpo_hparams[_dpo_hparam_name] = _dpo_hparam_value

            # load proteome tree if necessary
            proteome_file_name = dpo_hparams['proteome_file_name']
            if proteome_file_name is not None and \
            proteome_file_name not in ModelManager.proteome_trees:
                proteome_hash, proteome_tree = load_proteome_tree(proteome_file_name)
                ModelManager.proteome_trees[proteome_file_name] = proteome_tree

            self.dpo_hparams = dpo_hparams
            d_for_hash = {
                k: self.dpo_hparams[k] 
                for k in ModelManager.dpo_hparams_names 
                if (k in self.dpo_hparams and self.dpo_hparams[k] is not None)
            }
            self.id = dict_to_hash(d_for_hash, truncate=8)

        # the directory where the model is stored
        self.path = os.path.join(G.ENV.ARTEFACTS, 'models', self.id)
        # save dpo-hparams to the model
        if not self.is_base_model:
            dpo_hparams_yaml_file_path = join(self.path, "dpo_hparams.yaml")
            if not os.path.exists(dpo_hparams_yaml_file_path):
                self.dpo_hparams.to_yaml(dpo_hparams_yaml_file_path)

        # data
        self.dpo_structure_data_loaders = {}
        self.dpo_preference_data_loaders = {}

        # model parameters
        self.torch = None
        self.last_dpo_epoch = 0
        if load_ckpt_id is not None:
            _ckpt = CapeMPNN.from_file(os.path.join(self.path, "ckpts", f"{ckpt}.pt"))
            _ckpt.eval()
            _ckpt.to("cpu")
            self.last_dpo_epoch = int(ckpt.split("_")[1])
            self.torch = _ckpt

        # training data
        self.dpo_optimizer = None
        self.dpo_tuning_steps = 0
        self.dpo_losses = {split: [] for split in Split}
        
        self.models.update({self.id: self})

    def get_preference_scores_file_path(self):
        return join(self.path, "metrics", "preference_scores.tsv")

    def get_losses_file_path(self):
        return join(self.path, "metrics", "losses.tsv")
