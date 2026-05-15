import os
from enum import Enum

import numpy as np
import pandas as pd

from rapidfuzz.distance import Levenshtein

from kit.loch.utils import get_seq_hash
from kit.loch.path import get_pdb_file_path
from kit.bioinf.alignment.structure.tm_align import align_structures
from kit.bioinf.proteins import Protein
from kit.bioinf.proteins.eval.destress import load_destress_csv
from kit.bioinf.alignment.sequence import PairwiseSequenceAligner


db = None
nr_specific_trials = None
pdbs_to_exclude = None


def set_config(_db, _nr_specific_trials, _pdbs_to_exclude):
    global db, nr_specific_trials, pdbs_to_exclude
    db = _db
    nr_specific_trials = _nr_specific_trials
    pdbs_to_exclude = _pdbs_to_exclude


def reduce_dict(dictionary, source_ids, protein_ids):
    return {ks: {kp: v for kp, v in proteins.items() if kp in protein_ids} for ks, proteins in dictionary.items() if ks in source_ids}


def calc_TM_data(seq_hash_data, seq_hash_generated, missing=None):
    data_pdb_file_path = get_pdb_file_path(seq_hash_data, predictor_structure_name="exp")
    generated_pdb_file_path = get_pdb_file_path(seq_hash_generated)

    if os.path.exists(data_pdb_file_path) and os.path.exists(generated_pdb_file_path):
        _df = db.get_tm_align(seq_hash_data, seq_hash_generated)
        if len(_df) == 1:
            result = _df.iloc[0][['score', 'alignment_length', 'rmsd', 'identical', 'len_chain_1', 'len_chain_2']]
        elif len(_df) == 0:
            tm_score, tm_alignment_length, tm_rmsd, tm_identical, len_chain_1, len_chain_2 = align_structures(
                data_pdb_file_path, generated_pdb_file_path, return_chain_lengths=True)
            db.add_tm_align(seq_hash_data, seq_hash_generated, tm_score, tm_alignment_length, tm_rmsd, tm_identical, len_chain_1, len_chain_2)
            result = (tm_score, tm_alignment_length, tm_rmsd, tm_identical, len_chain_1, len_chain_2)
        else:
            raise Exception("Too many TM align results in database")
    else:
        result = missing

    return result


def calc_TM_score(seq_hash_data, seq_hash_generated, missing=None):
    result = calc_TM_data(seq_hash_data, seq_hash_generated, None)
    return result[0] if result is not None else missing


class BestTrial(Enum):
    MAX_TM = 0  # the trial with the maximum TM-score
    TM_NINETY_MIN_VIS = 1  # the trial with the loweset visibility of all trials with a TM-scores above 0.9 otherwise the max TM-score trial 

def transform_value(value, value_data, info_attr, delta=False, relative=False, missing=None):
    if not delta and not relative:
        return value

    if value is None or value_data is None:
        return missing

    if info_attr == 'dssp':
        return Levenshtein.distance(value, value_data) / (0.5 * (len(value) + len(value_data)))

    if delta:
        value -= value_data

    if relative:
        value /= value_data

    return value


destress_infos = {
    # info_attr: csv column name
    'dssp': 'dssp_assignment',
    'rosetta_total': 'rosetta_total',
    'hydrophobic_fitness': 'hydrophobic_fitness',
    'packing_density': 'packing_density',
    'aggrescan3d_avg': 'aggrescan3d_avg_value',
    'aggrescan3d_max': 'aggrescan3d_max_value',
    'dfire2_total': 'dfire2_total',
    'evoef2_total': 'evoef2_total',
    'isoelectric_point': 'isoelectric_point',
    'charge': 'charge',
    'mass': 'mass',
}

class Design:
    designs = {}
    aligner = PairwiseSequenceAligner(open_gap_score=-20, extend_gap_score=-10)
    
    def __init__(self, source_id, protein_id_or_data):
        protein_id = protein_id_or_data if not isinstance(protein_id_or_data, Design) else protein_id_or_data.data
            
        if source_id not in Design.designs:
            Design.designs[source_id] = {}
        assert protein_id not in Design.designs[source_id]
        
        Design.designs[source_id][protein_id] = self

        self.source_id = source_id
        self.data = protein_id_or_data  # either the protein_id (for a data Design) or a reference to the data Design

        # the trial data
        self.seqs = {}  # self.seqs[seq_hash] = seq
        self.designed_positions = {}  # self.designed_positions[seq_hash] = designed_positions
        self.tm_scores = {}  # self.tm_scores[seq_hash] = tm_score
        self.tm_data = {}  # self.tm_data[seq_hash] = (tm_score, tm_alignment_length, tm_rmsd, tm_identical)
        self.vis_mhc_1 = {}  # self.vis_mhc_1[(mhc_1_setup_hash, mhc_1_predictor_hash, proteome_hash)][seq_hash] = vis_mhc_1
        self.vis_mhc_1_abs = {}  # self.vis_mhc_1_abs[(mhc_1_setup_hash, mhc_1_predictor_hash, proteome_hash)][seq_hash] = vis_mhc_1
        self.blosum = {}

        # add DESTRESS attributes
        for info_attr in destress_infos.keys():
            setattr(self, info_attr, {})

    def rm_seq_hash(self, seq_hash):
        for info_attr in list(destress_infos.keys()) + ['seqs', 'designed_positions', 'tm_scores', 'tm_data', 'vis_mhc_1', 'vis_mhc_1_abs', 'blosum']:
            if seq_hash in getattr(self, info_attr):
                del getattr(self, info_attr)[seq_hash]

    @classmethod
    def get(cls, source_id, ref_design_or_protein_id):
        design = None
        if source_id in Design.designs:
            protein_id = ref_design_or_protein_id.data if isinstance(ref_design_or_protein_id, Design) else ref_design_or_protein_id
            if protein_id in Design.designs[source_id]:
                design = Design.designs[source_id][protein_id]

        if design is None:
            design = Design(source_id, ref_design_or_protein_id)
        return design

    @classmethod
    def load_info(cls, source_ids, protein_ids, info_name, *args, **kwargs):
        missing = []
        for source_id in source_ids:
            for protein_id in protein_ids:
                if source_id not in Design.designs or protein_id not in Design.designs[source_id]:
                    missing.append((source_id, protein_id))
                    continue
                getattr(cls.designs[source_id][protein_id], f"calc_{info_name}")(*args, **kwargs)
        return missing

    @classmethod
    def load_tm_scores(cls, source_ids, protein_ids):
        return cls.load_info(source_ids, protein_ids, 'tm_scores')

    @classmethod
    def load_tm_data(cls, source_ids, protein_ids):
        return cls.load_info(source_ids, protein_ids, 'tm_data')

    @classmethod
    def load_vis_mhc_1(cls, source_ids, protein_ids, mhc_1_setup, mhc_1_predictor, proteome_hash):
        mhc_1_predictor_hash = mhc_1_predictor.get_predictor_hash()
        qry = f"SELECT * FROM visibility_mhc_1 WHERE predictor_hash == '{mhc_1_predictor_hash}' AND "
        qry += (" proteome_hash IS NULL " if proteome_hash is None else f" proteome_hash == '{proteome_hash}' ")
        df_vis_mhc_1 = db.sql_to_df(qry)
        return cls.load_info(source_ids, protein_ids, 'vis_mhc_1', mhc_1_setup, mhc_1_predictor, proteome_hash, df_vis_mhc_1)

    @classmethod
    def load_destress(cls, source_ids, protein_ids, destress_dir_path=None):
        if destress_dir_path is not None:
            df_destress = load_destress_csv(destress_dir_path)
            cls.load_info(source_ids, protein_ids, 'destress', df_destress)
        else:
            cls.load_info(source_ids, protein_ids, 'destress')

    @classmethod
    def get_best_trials(cls, source_ids=None, protein_ids=None,
                        best_trial=BestTrial.MAX_TM, designed_positions='all', list_leaves=False):
        best = {}
        for source_id, proteins in cls.designs.items():
            if source_ids is None or source_id in source_ids:
                best[source_id] = {}
                for protein_id, design in proteins.items():
                    if protein_ids is None or protein_id in protein_ids:
                        design = cls.designs[source_id][protein_id]

                        best_seq_hashes = design.get_best_seq_hashes(
                            best_trial=best_trial,
                            designed_positions=designed_positions
                        )

                        if list_leaves:
                            best[source_id][protein_id] = best_seq_hashes
                        else:
                            assert len(best_seq_hashes) == 1
                            best[source_id][protein_id] = best_seq_hashes[0]

        return best

    def get_best_seq_hashes(self, best_trial=BestTrial.MAX_TM, designed_positions='all'):
        result = None
        if self.source_id.startswith('data'):
            return list(self.seqs)

        if designed_positions is None:
            seq_hashes = list(self.seqs.keys())
        else:
            seq_hashes = self.get_seq_hashes_for_designed_positions(designed_positions)

        max_tm_score, max_seq_hash = -1., None
        min_vis_mhc_1_abv, min_vis_mhc_1_abv_seq_hash = None, None

        for seq_hash in seq_hashes:
            _tm_score = self.tm_scores[seq_hash]
            if _tm_score is not None and _tm_score > max_tm_score:
                max_seq_hash = seq_hash
                max_tm_score = _tm_score

                if best_trial == BestTrial.TM_NINETY_MIN_VIS and _tm_score >= 0.9:
                    if min_vis_mhc_1_abv is None or self.vis_mhc_1[best_trial.mhc_1_key][
                        seq_hash] < min_vis_mhc_1_abv:
                        min_vis_mhc_1_abv = self.vis_mhc_1[best_trial.mhc_1_key][seq_hash]
                        min_vis_mhc_1_abv_seq_hash = seq_hash

        if best_trial == BestTrial.MAX_TM:
            result = max_seq_hash
        elif best_trial == BestTrial.TM_NINETY_MIN_VIS:
            result = max_seq_hash
            if min_vis_mhc_1_abv_seq_hash is not None:
                result = min_vis_mhc_1_abv_seq_hash

        return [result]

    def get_info(self, info_attr, seq_hash=None, key=None, delta=False, relative=False, missing=None):
        if seq_hash is None:
            result = []
            for seq_hash in list(self.seqs.keys()):
                result.append(self.get_info(info_attr, seq_hash, key, delta, relative, missing))
            return result

        result = missing
        info = getattr(self, info_attr)
        if key is not None:
            info = info[key]

        if seq_hash in info:
            if delta or relative:
                info_data = self.get_info_data(info_attr, key=key)
                result = transform_value(info[seq_hash], info_data, info_attr, delta, relative, missing)

            else:
                result = info[seq_hash]

        return result


    @classmethod
    def get_best_info(cls, best, info_attr, key=None, delta=False, relative=False, missing=None):
        results = {}
        for source_id, proteins in best.items():
            results[source_id] = {}
            for protein_id, entry in proteins.items():
                _res = missing
                if entry is not None:
                    design = cls.designs[source_id][protein_id]

                    # the leaves of best can either be lists of or single seq hashes
                    seq_hashes = [entry] if isinstance(entry, str) else entry
                    value = []
                    for seq_hash in seq_hashes:
                        value.append(design.get_info(info_attr, seq_hash, key, delta, relative, missing))

                    _res = value[0] if isinstance(entry, str) else np.array(entry)
                results[source_id][protein_id] = _res

        return results

    @classmethod
    def get_best_tm_scores(cls, best, missing=0):
        return cls.get_best_info(best, 'tm_scores', missing=missing)

    @classmethod
    def get_best_mhc_1_vis(cls, best, mhc_1_setup, mhc_1_predictor, proteome_hash):
        mhc_1_predictor_hash = mhc_1_predictor.get_predictor_hash()
        key = (mhc_1_setup, mhc_1_predictor_hash, proteome_hash)

        mhc_1_vis = cls.get_best_info(best, 'vis_mhc_1', key=key)
        mhc_1_vis_abs = cls.get_best_info(best, 'vis_mhc_1_abs', key=key)

        return mhc_1_vis, mhc_1_vis_abs

    def add_trial(self, seq, designed_positions):
        seq_hash = get_seq_hash(seq)
        self.seqs[seq_hash] = seq
        if seq_hash in self.designed_positions:
            self.designed_positions[seq_hash].add(designed_positions)
        else:
            self.designed_positions[seq_hash] = {designed_positions}
    
    def calc_tm_scores(self):
        self.calc_tm_data()
        for seq_hash, seq in self.seqs.items():
            if seq_hash in self.tm_data and self.tm_data[seq_hash] is not None:
                self.tm_scores[seq_hash] = self.tm_data[seq_hash][0]
            else:
                self.tm_scores[seq_hash] = None

    def calc_tm_data(self):
        if isinstance(self.data, str):
            for seq_hash, seq in self.seqs.items():
                self.tm_data[seq_hash] = (1., None, None, None, None, None)
        else:
            seq_hash_data = self.get_seq_hash_data()
            for seq_hash, seq in self.seqs.items():
                self.tm_data[seq_hash] = calc_TM_data(seq_hash_data, seq_hash, missing=None)

    def calc_vis_mhc_1(self, mhc_1_setup, mhc_1_predictor, proteome_hash, df_vis_mhc_1=None):
        mhc_1_predictor_hash = mhc_1_predictor.get_predictor_hash()
        key = (mhc_1_setup, mhc_1_predictor_hash, proteome_hash)
        if key not in self.vis_mhc_1:
            self.vis_mhc_1[key] = {}
            self.vis_mhc_1_abs[key] = {}

        if df_vis_mhc_1 is None:
            qry = f"SELECT * FROM visibility_mhc_1 WHERE predictor_hash == '{mhc_1_predictor_hash}' AND "
            qry += (" proteome_hash IS NULL " if proteome_hash is None else f" proteome_hash == '{proteome_hash}' ")
            df_vis_mhc_1 = db.sql_to_df(qry)
        
        if isinstance(self.data, str):  # data
            seq_hash = list(self.seqs.keys())[0]
            if not (key in self.vis_mhc_1 and seq_hash in self.vis_mhc_1[key]):
                qry = f"seq_hash == '{seq_hash}' and immuno_setup_mhc_1 == '{mhc_1_setup}' "
                self.vis_mhc_1[key][seq_hash] = 1.
                self.vis_mhc_1_abs[key][seq_hash] = int(df_vis_mhc_1.query(qry).visibility)
        else:
            data_seq_hash = self.get_seq_hash_data()

            self.data.calc_vis_mhc_1(mhc_1_setup, mhc_1_predictor, proteome_hash, df_vis_mhc_1)
            vis_mhc_1_abs_data = self.data.vis_mhc_1_abs[key][data_seq_hash]
            
            for seq_hash, seq in self.seqs.items():
                if seq_hash not in self.vis_mhc_1[key]:
                    qry = f"seq_hash == '{seq_hash}' and immuno_setup_mhc_1 == '{mhc_1_setup}' "
                    vis_mhc_1_abs = int(df_vis_mhc_1.query(qry).visibility)
    
                    self.vis_mhc_1[key][seq_hash] = vis_mhc_1_abs / vis_mhc_1_abs_data if vis_mhc_1_abs_data > 0 else None
                    self.vis_mhc_1_abs[key][seq_hash] = vis_mhc_1_abs

    def calc_destress(self, df_destress=None):
        for info_name, column_name in destress_infos.items():
            info = getattr(self, info_name)
            if df_destress is None:
                for seq_hash, seq in self.seqs.items():
                    value = None
                    if seq_hash in Protein.proteins:
                        value = Protein.proteins[seq_hash].get_info(info_name)

                    info[seq_hash] = value
            else:
                dtype = df_destress.dtypes[column_name]
                for seq_hash, seq in self.seqs.items():
                    if seq_hash in df_destress.index:
                        value = df_destress.loc[seq_hash, column_name]
                        if not isinstance(value, str) and np.isnan(value):
                            value = None
                        info[seq_hash] = value

    def calc_blosum(self):
        self.blosum = {}

        seq_data = self.get_info_data('seqs')
        chains_data = seq_data.split('/')
        for seq_hash, seq in self.seqs.items():
            chains = seq.split('/')

            bs = None
            if len(chains_data) == 1:
                seq_data = chains_data[0]

                bs = 0
                for seq in chains:
                    bs += self.aligner.score_seq_to_seq(seq_data, seq)
            else:
                if len(chains_data) == len(chains):
                    bs = 0
                    for seq_data, seq in zip(chains_data, chains):
                        bs += self.aligner.score_seq_to_seq(seq_data, seq)

            self.blosum[seq_hash] = bs


    def get_seq_hashes_for_designed_positions(self, designed_positions):
        seq_hashes = []
        for seq_hash, dp in self.designed_positions.items():
            if designed_positions in dp:
                seq_hashes.append(seq_hash)
        return seq_hashes

    def get_design_data(self):
        if isinstance(self.data, str):
            return self
        else:
            return self.data.get_design_data()

    def get_seq_hash_data(self):
        design_data = self.get_design_data()
        _h = list(design_data.seqs)
        assert len(_h) == 1
        return _h[0]

    def get_info_data(self, info, key=None):
        design_data = self.get_design_data()
        _info = getattr(design_data, info)
        if key is not None:
            _info = _info[key]
        _h = list(_info.values())
        assert len(_h) == 1
        return _h[0]

    def print(self, seq_hash, infos_to_plot):
        protein_id = self.get_design_data().data

        design_similar =  Design.designs['data_similar'][protein_id]

        print(f"{'INFO':25s}  {'VALUE':>20s} {'DATA':>20s} {'RANGE':>40s}")
        for info, specification in infos_to_plot.items():
            info_name = specification[0]
            value, value_data, values_similar = None, None, []

            delta = True if info_name.startswith('delta') or info == 'dssp' else False

            text = f"{info_name:25s}: {'Missing':>20s}"
            _info = getattr(self, info)
            if seq_hash in _info:
                value_data = self.get_info_data(info)
                value = _info[seq_hash]

                if not (value_data is None or value is None):
                    _infos_similar = getattr(design_similar, info)
                    for seq_hash_similar, _value_similar in _infos_similar.items():
                        if _value_similar is not None:
                            if delta:
                                _value_similar = transform_value(_value_similar, value_data, info, True)
                            values_similar.append(_value_similar)

                    if delta:   # if a comparison to the data value gets plotted
                        value = transform_value(value, value_data, info, True)
                        value_data = None

                    text = f"{info_name:25s}: {value:20.2f}"
                    if not delta:
                        text += f" {value_data:20.2f}"
                    else:
                        text += f" {'':20s}"
                    if len(values_similar) > 0:
                        range = f"{np.min(values_similar):.2f} - {np.max(values_similar):.2f}"
                        text += f" {range:>40s}"

            print(text)


def get_infos_df(source_ids, protein_ids, best, infos, rename_source_ids=None):
    df_plot = None
    for info in infos:
        info_attr, info_name, info_delta, info_rel, info_normalize = info[:5]
        info_key = info[5] if len(info) == 6 else None

        best_values = Design.get_best_info(best, info_attr, delta=info_delta, relative=info_rel, key=info_key)

        sources, proteins, values = [], [], []
        for source_id in source_ids:
            for protein_id in protein_ids:
                value = None

                if not protein_id in Design.designs[source_id]:
                    sources.append(source_id)
                    proteins.append(protein_id)
                    values.append(None)
                    continue
                    
                design = Design.designs[source_id][protein_id]

                seq_hashes = None
                if source_id.startswith('data'):
                    seq_hashes = list(design.seqs.keys())
                else:
                    _best = best[source_id][protein_id]
                    if _best is not None:
                        seq_hashes = [_best]

                if seq_hashes is not None:
                    for i, seq_hash in enumerate(seq_hashes):
                        value = design.get_info(info_attr, seq_hash=seq_hash, key=info_key, delta=info_delta, relative=info_rel)
                        scale = 1.
                        if info_normalize:  # normalize by the count of amino acids in the protein
                            if seq_hash in design.dssp and design.dssp[seq_hash] is not None:
                                scale = 1./len(design.dssp[seq_hash])
                            else:
                                value = None

                        if rename_source_ids is not None and source_id in rename_source_ids:
                            source_id = rename_source_ids[source_id]
                        _source_id = f"{source_id}@{i}" if len(seq_hashes) > 1 else source_id
                        sources.append(_source_id)
                        proteins.append(protein_id)
                        value = value*scale if value is not None else None
                        values.append(value)
                else:
                    sources.append(source_id)
                    proteins.append(protein_id)
                    values.append(None)

        df = pd.DataFrame({
            'source': sources,
            'protein': proteins,
            info_name: values
        })
        # values = reduce_dict(values, source_ids, protein_ids)
        # df = pd.DataFrame(values).reset_index()
        # df.rename(columns={'index': 'protein'}, inplace=True)
        # df = df.melt(id_vars=['protein'], var_name='source', value_name=info_name)
        df['index'] = df.apply(lambda r: f"{r.source}-{r.protein}", axis=1)
        df = df.set_index('index')

        if df_plot is None:
            df_plot = df
        else:
            df_plot = df_plot.join(df[[info_name]], how='outer')

    df_plot['epoch'] = df_plot.source.apply(lambda s: int(s.split("_")[-1]) if ':epoch_' in s else -1)

    return df_plot.sort_values(['protein', 'source'])
