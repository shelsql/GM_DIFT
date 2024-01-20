# Generative Models Final Project
## Image Classification with Diffusion Features

## Environment Setup
Ubuntu 20.04  CUDA Version: 12.2

``` bash
conda env create -f environment.yml
pip install tensorboardX scikit-learn
conda activate dift
```

## Quick Start

``` bash
git clone git@github.com:openai/guided-diffusion.git

bash get_adm.sh

python prepare_data.py --datadir path_to_cifar_10 --savedir dataset_parent_dir -t --up_ft_index --ensemble

python train.py --datadir path_to_extracted_features --epochs --lr

python test.py
```
