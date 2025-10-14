#!/usr/bin/env bash

conda activate base
conda env remove --name cape
conda env create -f ${PF}/setup/environment.yml

conda activate cape
echo "conda activate cape" >> ${HOME}/.bashrc

jupyter lab --generate-config
echo "c.ServerApp.ip = '0.0.0.0'" >> ${HOME}/.jupyter/jupyter_notebook_config.py
