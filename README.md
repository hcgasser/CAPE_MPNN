<center><img src="logo.jpg" alt="Cape Logo" width="200" height="200"><br><br><br></center>


# Controlled Amplitude of Present Epitopes (CAPE)

Proteins, due to their versatility and diverse production methods, have attracted substantial interest for both industrial and therapeutic applications.
The design of new therapeutics requires careful consideration of immune responses, particularly cytotoxic T-lymphocyte (CTL) reactions to intracellular proteins.
Computational protein design methods should therefore take these into account.

## CAPE-MPNN

This repository contains the code for **CAPE-MPNN**, one of the methods we developed to address this challenge.
It was introduced in the article *[Tuning ProteinMPNN to reduce protein visibility via MHC Class I through direct preference optimization](https://doi.org/10.1093/protein/gzaf003)*.

This method applies Direct Preference Optimization (DPO) to [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) to reduce the number of MHC Class I epitopes in generationed proteins.
The resulting fine-tuned model weights are referred to as **CAPE-MPNN**.

---

### Installation

This project's code is provided under the MIT License.
Some portions of the code are adapted from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), which is also MIT-licensed. The corresponding files include a copy of ProteinMPNN's full MIT license.

Users are responsible for obtaining and complying with the licenses of all third-party dependencies (see `external/README.md`).


#### General Requirements

The programs in this repository require a Linux machine with Docker and GPU support installed.
Also, for evaluation, a structure prediction tool (e.g. [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) as well as a local version of [DE-STRESS (headless, including Rosetta)](https://github.com/wells-wood-research/de-stress) need to be available.


#### Setup the container

To replicate the environment used in the preparation of the paper, we recommend using a Docker container, although this is not strictly required.

##### Clone the CAPE-MPNN repository

```bash
export REPO=CAPE_MPNN
git clone https://github.com/hcgasser/${REPO}.git
export CAPE=<path to repo folder>
```

##### Create the Docker image

```bash
cd $CAPE
source ${CAPE}/setup/create_image.sh    # set a container password, e.g. 'cape_pwd'
```

##### Finalize the container setup

* Start the container:

  ``docker run --privileged --name cape_mpnn_container --gpus all -it -p 9000:9000 -v ${CAPE}:/${REPO} cape_mpnn``
  
  in some cases adding the parameters `` --runtime nvidia --shm-size <sufficient shared memory e.g. 32g>`` might be necessary

* The container can be exited with ``exit`` and restarted with: ``docker start -i cape_mpnn_container``
* The option ``-v ${CAPE}:/${REPO}`` mounts the host folder ``${CAPE}`` to ``/CAPE_MPNN/`` inside the container, enabling data exchange
* ⚠ For licensing reasons, third-party software must be installed manually by the user inside the container (see `external/README.md`)


---

### Fine-tuning

First, we specify the MHC-I alleles to deimmunize against

```bash
export MHC_Is="HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01"
```


#### Setup PWM-based MHC-I presentation predictor

```bash 
MHC-I_rank_peptides.py \
    --output ${PF}/data/input/immuno/mhc_1/Mhc1PredictorPwm \
    --alleles ${MHC_Is} \
    --tasks rank+pwm+stats+agg \
    --peptides_per_length 1000000
```


#### DPO hyperparameter search

The configuration file `${PF}/configs/CAPE-MPNN/hyp/hyp_b69bb1.yaml` defines the hyperparameter ranges.
Parameters such as *batch_size* and *max_protein_length* (in residues) are especially important, as large values may cause GPU memory overflows.

```bash 
export HOSTNAME='workstation'
for i in {1..20}; do
    cape-mpnn.py --hyp ${PF}/configs/CAPE-MPNN/hyp/hyp_b69bb1.yaml --hyp_n 1
done
```

Each ``cape-mpnn.py`` run creates a fine-tuned **CAPE-MPNN** model. 
Models can be found on the host system in `${CAPE}/artefacts/CAPE-MPNN/models/<ModelID>`.

* `dpo_hparams.yaml` lists the fine-tuning hyperparameters
* The `ckpts/`  subfolder contains the resulting model weights, which can be used as replacements for **ProteinMPNN** weights in downstream tasks.

---

### Evaluation

This section describes how we obtained the results shown in the [paper](https://doi.org/10.1093/protein/gzaf003).

First the jupyter server needs to be started in the container:

```bash
tmux new-session -t jupyter
jupyter lab password    # set a jupyter server password, e.g. 'cape_pwd'
jupyter lab --port 9000
```

On the host system, open a browser and navigate to `http://localhost:9000/`.
Run the notebook `CAPE-Eval/cape-eval_mpnn.ipynb`

* Designs are stored in `${CAPE}/artefacts/CAPE-MPNN/outputs/fasta/all/`
* Files prepared for structure prediction are saved in `${CAPE}/artefacts/CAPE/colabfold/input/`
* ColabFold outputs are expected in `${CAPE}/artefacts/CAPE/colabfold/output/`
