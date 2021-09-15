
import torchvision.models as models
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
from torchvision.datasets import ImageFolder

from torchvision import transforms
from torchvision.transforms import ToTensor

train_transformations = transforms.Compose([
    transforms.Resize((256, 256)),  # resize input images to 255,255
    transforms.ToTensor()
])

test_transformations = transforms.Compose([
    transforms.Resize((256, 256)),  # resize input images to 255,255
    transforms.ToTensor()
])

# applt the train and test transformations
#training_dataset = ImageFolder(data_dir+'/Training', transform=train_transformations)
#testing_dataset= ImageFolder(data_dir+'/Testing', transform=test_transformations)
training_dataset = ImageFolder(
    root='E:/Places2/train', transform=train_transformations)
testing_dataset = ImageFolder(
    root='E:/Places2/test', transform=test_transformations)

device = torch.device("cpu")
# def load(f, map_location='cpu', pickle_module=pickle, **pickle_load_args):


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return training_dataset.classes[preds[0].item()]


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss


class Resnet34CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)

        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 2)

    def forward(self, xb):
        xb = self.network(xb)
        return xb

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


model_resnet34 = to_device(Resnet34CnnModel(), device)
model_resnet34.load_state_dict(torch.load(
    "C:/Users/Prasoon Smirti/Downloads/modelplacesusingresnet34.pth", map_location=torch.device('cpu')))
#use_cuda = torch.cuda.is_available()

img, label = testing_dataset[125]
plt.imshow(img.permute(1, 2, 0))
show(block=True)
print('Label:', training_dataset.classes[label], ', Predicted:', predict_image(
    img, model_resnet34))
