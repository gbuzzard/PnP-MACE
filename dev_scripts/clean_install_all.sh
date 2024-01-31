#!/bin/bash

# Create and activate new conda environment

ROOTPATH=$(realpath $(dirname $0)/..)

# First check if the target environment is active and deactivate if so
NAME=pnp_mace
if [ "$CONDA_DEFAULT_ENV"==$NAME ]; then
    conda deactivate
fi

# Then remove the old version and reinstall
conda remove env --name $NAME --all -y
conda create --name $NAME python=3.10 -y
conda activate $NAME

pip install -r $ROOTPATH/docs/requirements.txt
pip install $ROOTPATH

cd $ROOTPATH/docs
/bin/rm -rf _build

make clean html

echo
echo "Use 'conda activate" $NAME "' to activate this environment."
echo

echo ""
echo "*** The html documentation is at PnP-MACE/docs/_build/html/index.html ***"
echo ""
