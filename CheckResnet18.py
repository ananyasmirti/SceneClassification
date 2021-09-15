
from matplotlib.pyplot import show
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import time
import copy
import os
transforms = transforms.Compose(
    [
        transforms.ToTensor()
    ])
train_dataset = datasets.ImageFolder(
    root='E:/Places2/train', transform=transforms)
test_dataset = datasets.ImageFolder(
    root='E:/Places2/test', transform=transforms)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def load(f, map_location='cpu', pickle_module=pickle, **pickle_load_args):


net = torch.load('C:/Users/Prasoon Smirti/Downloads/modelResnet18.pth',
                 map_location=torch.device('cpu'))
net.eval()
use_cuda = torch.cuda.is_available()


def imshow(inp, title=None):

    inp = inp.cpu() if device else inp
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def visualize_model(net, num_images=10):
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))

    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy() if use_cuda else preds.numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(2, num_images//2, images_so_far)
            ax.axis('off')
            ax.set_title('predictes: {}'.format(
                test_dataset.classes[preds[j]]))
            imshow(inputs[j])

            if images_so_far == num_images:
                return


plt.ion()
visualize_model(net)
show(block=True)
# plt.ioff()
