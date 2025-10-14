import os
import shutil

import kit.globals as G
from kit.path import join
from kit.data import str_to_file
from kit.bioinf.pdb import download_structure, pdb_to_seqs, get_similar_structures
from kit.log import log_info
from kit.bioinf.utils import chains_to_seq, get_seq_hash
from kit.loch.path import cp_fasta_to_dir, cp_pdb_to_dir, get_pdb_file_path
from kit.bioinf.proteins import Protein

from CAPE.Eval.MPNN import E
from CAPE.Eval.MPNN.design import Design
from kit.bioinf.proteins import ProteinType


c_avg_vis_mhc_1_v_g = 'avg_vis_mhc_1_v_g'
c_avg_vis_mhc_1_pc_v_g = 'avg_vis_mhc_1_pc_v_g'
c_avg_rec_v_g = 'avg_rec_v_g'
c_avg_tm_v_s = 'avg_tm_v_s'
c_avg_vis_mhc_1_v_s = 'avg_vis_mhc_1_v_s'
c_avg_tm_t_s = 'avg_tm_t_s'
c_avg_vis_mhc_1_t_s = 'avg_vis_mhc_1_t_s'


def sources_to_model_ids(sources):
    return [x.split(':')[0] for x in sources]


def download_specific_protein(pdb_id, protein_type, pdb_dir_path, model_nr=0, source_id='data', similar_to=None,
                              add_to_db=True, add_to_loch=True, d_pdb_to_seqs_kwargs={},
                              file_format="pdb", source='pdb', server='https://files.rcsb.org', keep_chains={}):
    try:
        pdb_file_path = download_structure(
          pdb_id, 
          pdb_dir_path, 
          file_format=file_format, 
          source=source,
          server=server
        )

        protein = Protein.from_pdb(pdb_file_path, 
          keep_chains=keep_chains.get(pdb_id, None), 
          structure_to_seq_kwargs=d_pdb_to_seqs_kwargs.get(pdb_id, None),
          seq_translate=("-", "X", "")
        )

        # _pdb_to_seqs_kwargs = None
        # if d_pdb_to_seqs_kwargs is not None and pdb_id in d_pdb_to_seqs_kwargs:
        #     _pdb_to_seqs_kwargs = d_pdb_to_seqs_kwargs[pdb_id]
        #     models = pdb_to_seqs(pdb_file_path, return_full=True, **_pdb_to_seqs_kwargs)
        # else:
        #     models = pdb_to_seqs(pdb_file_path, return_full=True)
        # if len(models) > 1:
        #     log_info(f"Multiple models found for {pdb_id}. Using model {model_nr}.")
        # chains = models[model_nr]
        # if protein_type == ProteinType.MONOMER:
        #     assert len(chains) == 1
        # seq = chains_to_seq(chains)
        # seq = seq.replace("-", "X")  # ProteinMPNN will predict "X" here, to be consistent we also use this in the data

        # seq_hash = get_seq_hash(seq)

        seq = protein.get_seq()
        seq_hash = protein.seq_hash

        # add to memory
        pdb_id_or_data = similar_to if similar_to is not None else pdb_id
        Design.get(source_id, pdb_id_or_data).add_trial(seq, 'all')

        if add_to_db:
            # add to DB
            _seq_hash = E.DB.add_seq(seq)
            assert _seq_hash == seq_hash
            E.DB.add_seq_to_list('specific', source_id, pdb_id, 'all', 1, seq_hash)

        if add_to_loch:
            protein.to_loch(predictor_structure_name='exp')
            # # add to loch
            # loch.add_entry(
            #   seq=seq, 
            #   pdb_id=pdb_id, 
            #   pdb_file_path=pdb_file_path, 
            #   model_nr=model_nr,
            #   pdb_to_seqs_kwargs=_pdb_to_seqs_kwargs
            # )

        return seq, seq_hash, protein
    except Exception as e:
        print(f"Error {pdb_id}: {e}")
        return None, None, None


def send_to_pdb(source_ids, protein_ids, designed_positions_name, colabfold_path, min_AAs_diversity=15, max_AA_content=0.5):
    cnt = 0
    to_pdb_dir_path = os.path.join(colabfold_path, 'input')
    for protein_id in protein_ids:
        for source_id in source_ids:
            design = Design.designs[source_id][protein_id]
            seq_hashes = [h for h, dpn in design.designed_positions.items() if designed_positions_name in dpn]
            for seq_hash in seq_hashes:
                seq = design.seqs[seq_hash]

                counts = {}
                for aa in seq:
                  counts[aa] = counts.get(aa, 0)+1

                _max_AA_content = max(counts.values()) / len(seq)

                if len(set(seq)) >= min_AAs_diversity and _max_AA_content <= max_AA_content:
                    fasta_file_path = os.path.join(to_pdb_dir_path, f"{seq_hash}.fasta")
                    if not os.path.exists(fasta_file_path):
                        cnt += 1
                    cp_fasta_to_dir(seq_hash, to_pdb_dir_path, translate=("/", ":", ""))
    print(f"Structures to predict: {cnt}")


def cp_pdb_to_dir_path(source_ids, protein_ids, tgt_dir_path, predictor_structure_name='AF', designed_positions=None):
    c_found, c_copied, missing = 0, 0, []
    missing_designs = []
    for protein_id in protein_ids:
        for source_id in source_ids:
            if source_id in Design.designs and protein_id in Design.designs[source_id]:
                design = Design.designs[source_id][protein_id]
                for seq_hash in design.seqs.keys():
                    if designed_positions is None or designed_positions in design.designed_positions[seq_hash]:
                        c_found += 1
                        res = cp_pdb_to_dir(seq_hash, tgt_dir_path, predictor_structure_name=predictor_structure_name)
                        if res != True:
                            if res == False:
                                missing.append(seq_hash)
                            else:
                                c_copied += 1
            else:
                missing_designs.append((source_id, protein_id))

    print(f"Found/copied/missing: {c_found}/{c_copied}/{len(missing)}\n{missing_designs}")
    return missing



def write_config_yaml(
        details,
        pod_argument
):
    config_dir_path = os.path.join(G.ENV.PROJECT, "configs", "CAPE-MPNN", "hyp")

    config_file_path = os.path.join(config_dir_path, f"{pod_argument}.yaml")
    config_file = f'''---
 COMMAND: "cape-mpnn.py"
 HYPERPARAMETERS:
  mhc_1_alleles:
   KIND: "fixed:str"
   VALUE: "{details.mhc_1_alleles}"
  base_model_name:
   KIND: "fixed:str"
   VALUE: "{details.base_model_name}"
  beta:
   KIND: "fixed:float"
   VALUE: {details.beta}
  reload_data_every_n_epochs:
   KIND: "fixed:int"
   VALUE: {details.reload_data_every_n_epochs}
  temperature:
   KIND: "fixed:float"
   VALUE: {details.temperature}
  num_epochs:
   KIND: "fixed:int"
   VALUE: {details.epoch}
  lr:
   KIND: "fixed:float"
   VALUE: {details.lr}
  num_examples_per_epoch:
   KIND: "fixed:int"
   VALUE: {details.num_examples_per_epoch}
  batch_size:
   KIND: "fixed:int"
   VALUE: {details.batch_size}
  max_protein_length:
   KIND: "fixed:int"
   VALUE: {details.max_protein_length}
  mhc_1_predictor:
   KIND: "fixed:str"
   VALUE: "{details.mhc_1_predictor}" '''

    if 'preference_sampling_method' in details:
        config_file += f'''
  preference_sampling_method:
   KIND: "fixed:str"
   VALUE: "{details.preference_sampling_method}"'''

    config_file += f'''
  rescut:
   KIND: "fixed:float"
   VALUE: {details.rescut}
  job:
   KIND: "env:str"
   NAME: "HOSTNAME"
   EVAL: "lambda x: x.split('-')[-1]"
'''
    str_to_file(config_file, config_file_path)


def write_job_yaml(pod_argument):
    pod_argument_hyphen = pod_argument.replace("_", "-")

    kube_job_dir_path = os.path.join(G.ENV.PROJECT, 'kube', 'CAPE-MPNN')

    kube_job_file_path = os.path.join(kube_job_dir_path, f"job_{pod_argument}.yaml")
    kube_job_file = f'''
apiVersion: batch/v1
kind: Job
metadata:
  name: s2118339-{pod_argument_hyphen}
  labels:
    app: s2118339-{pod_argument_hyphen}
    eidf/user: s2118339
    kueue.x-k8s.io/queue-name: informatics-user-queue
  annotations:
    description: "h.gasser@sms.ed.ac.uk"
spec:
  parallelism: 1
  completions: 1
  template:
    metadata:
      name: s2118339-{pod_argument_hyphen}
      labels:
        app: s2118339-{pod_argument_hyphen}
        eidf/user: s2118339
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-80GB'
      containers:
      - name: main
        image: ghcr.io/hcgasser/cape_kube:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: 48
            memory: 96000Mi
          limits:
            nvidia.com/gpu: 1
            cpu: 48
            memory: 96000Mi
        env:
        - name: REPO
          value: "CAPE"
        - name: ARGS
          value: "{pod_argument}"
        - name: SECRETS
          valueFrom:
            secretKeyRef:
              name: s2118339-secrets
              key: SECRETS
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
      imagePullSecrets:
      - name: s2118339-ghcr-secret
    '''

    str_to_file(kube_job_file, kube_job_file_path)
