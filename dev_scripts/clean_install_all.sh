#!/bin/bash

# Create and activate new conda environment
# First check if the target environment is active and deactivate if so
NAME=pnp_mace
if [ "$CONDA_DEFAULT_ENV"==$NAME ]; then
    conda deactivate
fi

# Then remove the old version and reinstall
yes | conda remove env --name $NAME --all
yes | conda create --name $NAME python=3.9
conda activate $NAME
pip install -r ../requirements.txt
pip install ..

pip install -r ../demo/requirements_demo.txt

cd ../docs
/bin/rm -r build

pip install -r requirements.txt
make clean html

echo ""
echo "*** The html documentation is at pcdrecon/docs/build/html/index.html ***"
echo ""

cd ../dev_scripts
echo
echo "Use 'conda activate" $NAME "' to activate this environment."
echo
