# export HF_ENDPOINT=https://hf-mirror.com
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from tqdm import trange
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer
from src.models.dift_adm import ADMFeaturizer

import argparse

category = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

########################## png #########################################
def download_data(data_path):
    
    train_data = torchvision.datasets.CIFAR10(data_path,  train=True, download=True, transform=None)
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=None)

    train_len = len(train_data)
    test_len = len(test_data)

    for i in trange(train_len):
        split_path = os.path.join(data_path, "train")
        os.makedirs(split_path, exist_ok=True)
        image,label = train_data[i][0], train_data[i][1]
        file_path = os.path.join(split_path, f"{i:0>10}_{label}.png")
        image.save(file_path)

    for i in trange(test_len):
        split_path = os.path.join(data_path, "test")
        os.makedirs(split_path, exist_ok=True)
        image,label = test_data[i][0], test_data[i][1]
        file_path = os.path.join(split_path, f"{i:0>10}_{label}.png")
        image.save(file_path)

def get_SD_feature(data_path = "./dataset_sd"):
    dift = SDFeaturizer('./huggingface')
    
    for split in ['train', 'test']:
        split_path = os.path.join(data_path, split)   
        X, Y = [], []
        for name in tqdm(os.listdir(split_path)):
            img = Image.open(os.path.join(split_path, name)).convert('RGB')
            img = img.resize((32,32))
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
            y = int(name[-5])
            ft = dift.forward(
                    img_tensor,                                 # 原始的图片
                    prompt=f'a photo of a {category[y]}',       # 生成模型用到的prompt
                    t=261,                                      # 
                    up_ft_index=0, 
                    ensemble_size=2
                )
            
            X.append(ft.squeeze(0).cpu().reshape(-1)) # [c, h, w]
            Y.append(y)
        
        X = torch.stack(X)      # (n, 1280)
        print(X.shape)
        Y = torch.IntTensor(Y)  # (n,)
        print(Y.shape)
        torch.save((X,Y), os.path.join(data_path, f'{split}.pt'))

def get_ADM_feature(data_path = "./dataset_adm"):
    dift = ADMFeaturizer()
    
    for split in ['test', 'train']:
        split_path = os.path.join(data_path, split)   
        X, Y = [], []
        for name in tqdm(os.listdir(split_path)):
            img = Image.open(os.path.join(split_path, name)).convert('RGB')
            img = img.resize((256,256))
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
            y = int(name[-5])
            ft = dift.forward(
                    img_tensor,                                 # 原始的图片
                    t=101,                                      # 
                    up_ft_index=0, 
                    ensemble_size=2
                )
            #print(ft.shape)# 1, 1024, 8, 8
            
            X.append(ft.squeeze(0).cpu()) #1024, 8, 8
            Y.append(y)
        
        X = torch.stack(X)      # (n, 1024, 8, 8)
        print(X.shape)
        Y = torch.IntTensor(Y)  # (n,)
        print(Y.shape)
        torch.save((X,Y), os.path.join(data_path, f'{split}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, default="sd")
    parser.add_argument("--datadir", type=str, default="./dataset")
    #parser.add_argument("--download", type=bool, defualt=True)
    args = parser.parse_args()
    data_path = args.datadir
    #download_data(data_path)
    if args.model == "sd":
        get_SD_feature(data_path)
    else:
        get_ADM_feature(data_path)
    x,y = torch.load(os.path.join(data_path, 'train.pt'))
    print(x.shape, y.shape)
    x,y = torch.load(os.path.join(data_path, 'test.pt'))
    print(x.shape, y.shape)
