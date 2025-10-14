#!/bin/bash

read -p 'Set container password (e.g. cape_pwd): ' password

pushd $CAPE

mkdir -p external/repos
mkdir -p external/data/ProteinMPNN
mkdir -p external/bioinf/netMHCpan
mkdir -p external/bioinf/TMalign
mkdir -p data/input/immuno/mhc_1/Mhc1PredictorPwm
mkdir -p artefacts/CAPE-MPNN/loch/structures/AF/pdb

docker build --build-arg REPO=${REPO} --build-arg PASSWORD=${password} -f setup/Dockerfile -t cape_mpnn .
popd
