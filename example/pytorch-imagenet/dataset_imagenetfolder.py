import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config



def make_imagenetfolder_loader(batch_size, num_workers=2, data_root=config.docker_imagenet_folder_root, train=True, val=True, pin_memory=True):
    print("Building ImageNet Folder data loader with {} workers".format(num_workers))
    ds = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        train_dataset = torchvision.datasets.ImageFolder(root=(data_root+'train/'), 
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,]), 
            target_transform=None,)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory)
        ds.append(train_loader)

    if val:
        val_dataset = torchvision.datasets.ImageFolder(root=(data_root+'val/'), 
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,]), 
            target_transform=None,)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory)
        ds.append(val_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds

    
  