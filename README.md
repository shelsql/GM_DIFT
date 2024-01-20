# Generative Models Final Project
## Image Classification with Diffusion Features

By 梁世谦，时有恒，刘知一，周正

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


### 分工

梁世谦：代码实现，运行实验，撰写报告

时有恒：代码实现，运行实验

刘知一：代码实现，结果可视化

周正：代码实现，结果可视化