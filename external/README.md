# Third party software

Check that you obtain and comply with all the licenses for the external software and packages that our tool relies on.
In particular:

- The ``continuumio/miniconda3`` Docker image
- The programs installed by ``setup/Dockerfile``
- NetMHCpan-4.1b
- TMalign
- [localcolabfold](https://github.com/YoshitakaMo/localcolabfold)
- [DE-STRESS](https://github.com/wells-wood-research/de-stress)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- all packages listed in ``setup/environment.yml``

## Host system

**CAPE-MPNN** calls functions from the **ProteinMPNN** [repo](https://github.com/dauparas/ProteinMPNN). 
It has to be cloned into the folder ``${CAPE}/external/repos/ProteinMPNN``.

We also used their 18 GB large [data file](https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz) in the fine-tuning process.
To replicate the fine-tuning process this needs to be downloaded and its content extracted into the folder ``${CAPE}/external/data/ProteinMPNN/pdb_2021aug02``.

If the user plans to run the evaluation workflow, [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) and [DE-STRESS](https://github.com/wells-wood-research/de-stress) need to be installed on the host system.
Check that the command ``colabfold_batch`` can be run from any path on the host system.


## Container

At first, the conda environment 'cape' needs to be set up

- check that you comply with all the licenses of the dependencies in ``setup/environment.yml``
- run: ``. ${PF}/setup/setup_conda.sh``


To install NetMHCpan and TMalign in the container, obtain the necessary software and data as well as **all required licences**:

- NetMHCpan: copy ``netMHCpan-4.1b.Linux.tar.gz`` and ``data.tar.gz`` from ``https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/`` into ``${CAPE}/external/bioinf/netMHCpan/`` on the host system
- TMalign: copy ``TMalign.cpp`` and ``readme.c++.txt`` from ``https://zhanggroup.org/TM-align/`` into ``${CAPE}/external/bioinf/TMalign/`` on the host system


The folder ``${CAPE}/external/bioinf`` should then have the following structure:

```
bioinf
├── netMHCpan
│   ├── data.tar.gz
│   └── netMHCpan-4.1b.Linux.tar.gz
└── TMalign
    ├── readme.c++.txt
    └── TMalign.cpp
```

Then run the following in the container:

# TMalign

After checking the licences run the following commands to install TMalign. Afterwards, check that the command ``TMalign`` is working in any folder within the container.

```shell
#
# remove previous installation
folder=${PROGRAMS}/TMalign
rm -rf $folder
mkdir $folder
pushd ${SOFTWARE}/bioinf/TMalign
#
# copy source code
cp TMalign.cpp $folder/
#
# compile program
pushd $folder
g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
#
# return to folder
popd
popd
#
# create link to executable
rm -f ${PROGRAMS}/bin/TMalign
ln -s ${folder}/TMalign ${PROGRAMS}/bin/TMalign
```

# NetMHCpan

After checking the licences run the following commands to install NetMHCpan.  Afterwards, check that the command ``netMHCpan`` is working in any folder within the container.

```shell
#
# remove previous installation
pushd ${SOFTWARE}/bioinf/netMHCpan
rm -rf ${PROGRAMS}/netMHCpan
mkdir ${PROGRAMS}/netMHCpan
#
# copy installation files
cp data.tar.gz ${PROGRAMS}/netMHCpan/
cp netMHCpan-4.1b.Linux.tar.gz ${PROGRAMS}/netMHCpan/
#
# unzip data
pushd ${PROGRAMS}/netMHCpan
tar -xzvf netMHCpan-4.1b.Linux.tar.gz
rm netMHCpan-4.1b.Linux.tar.gz
tar -xzvf data.tar.gz -C ./netMHCpan-4.1
rm data.tar.gz
#
# setup configuration file
pushd netMHCpan-4.1
progs=$(echo "$PROGRAMS" | sed 's/\//\\\//g')
replace="s/\/net\/sund-nas.win.dtu.dk\/storage\/services\/www\/packages\/netMHCpan\/4.1\/netMHCpan-4.1/${progs}\/netMHCpan\/netMHCpan-4.1/"
sed -i "$replace" netMHCpan
sed -i "s/TMPDIR  \/tmp/TMPDIR  \$NMHOME\/tmp/" netMHCpan
mkdir ./tmp
popd
popd
popd
#
# create symbolic link
rm -f ${PROGRAMS}/bin/netMHCpan
ln -s ${PROGRAMS}/netMHCpan/netMHCpan-4.1/netMHCpan ${PROGRAMS}/bin/netMHCpan
```

