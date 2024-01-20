from PIL import Image
from torchvision.transforms import PILToTensor
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.models.dift_sd import SDFeaturizer
from src.models.dift_adm import ADMFeaturizer
import cv2
import numpy as np
import argparse
import os

def extract_features(input_path, img_size = [256,256], prompt='', t=101, up_ft_index=0, ensemble_size=2):
    dift = ADMFeaturizer()
    img = Image.open(input_path).convert('RGB')
    img = img.resize(img_size)
    #img = remove_background(img)
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    ft = dift.forward(img_tensor,
                      t=t,
                      up_ft_index=up_ft_index,
                      ensemble_size=ensemble_size)
    upsample_layer = torch.nn.Upsample(size=(32, 32), mode='nearest')
    ft = upsample_layer(ft)
    print(ft.shape)
    return ft.squeeze(0).cpu()

def visualize_features(features, img_size, output_path):
    X = features.permute(1, 2, 0).reshape(-1, features.shape[0])
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    X_pca = (X_pca - X_pca.min(axis=0)) / (X_pca.max(axis=0) - X_pca.min(axis=0))
    img_pca = X_pca.reshape(features.shape[1], features.shape[2], 3)

    # Save the image
    plt.imsave(output_path, img_pca)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--save_path", type=str, default="output")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    features = extract_features(args.image_path)
    image_name = args.image_path.split("/")[-1][:-4]
    visualize_features(features, [32,32],os.path.join(args.save_path, image_name + "_features.jpg"))
    img = Image.open(args.image_path)
    img.save(os.path.join(args.save_path, args.image_path.split('/')[-1]))