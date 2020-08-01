from __future__ import print_function

import copy
import os.path as osp

import click
import sys
import cv2
import os
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from models import Ghostnet_TL, Ghostnet_proto
from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)

def get_files_list(images_path):
    images_path_list = [i for i in os.listdir(images_path) if i.endswith('.png') or i.endswith('.jpg')]
    images_path_list = [os.path.join(images_path,i) for i in images_path_list]
    return images_path_list

def apply_gradcam(image_paths, target_layer, output_dir, cuda=0):
    target_class = 1.0
    device = get_device(cuda)
    model = Ghostnet_proto()
    model.to(device)
    model.eval()
    images_plist = get_files_list(image_paths)
    images, raw_images = load_images(images_plist)
    images = torch.stack(images).to(device)
    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.FloatTensor([[target_class]] * len(images)).to(device)
    #print(ids_)
    gcam.backward(ids=ids_)
    regions = gcam.generate(target_layer=target_layer)
    for j in range(len(images)):
        print(
            "\t#{}: {} ({:.5f})".format(
                j, 1, float(probs)
            )
        )

        save_gradcam(
            filename=osp.join(
                output_dir,
                "{}-{}-gradcam-{}-{}.png".format(
                    j, "ghostnet", target_layer, 1
                ),
            ),
            gcam=regions[j, 0],
            raw_image=raw_images[j],
        )




if __name__ == '__main__':
    image_dir = 'test_samples/'
    target_layer = 'vanilla_ghostnet.conv_stem'
    output_dir = 'results/'
    apply_gradcam(image_dir, target_layer, output_dir)
