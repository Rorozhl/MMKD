"""
get data loaders
"""
from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms
from .cub2011 import Cub2011, Cub2011Sample
from .dogs import Dogs, DogsSample
from .mit67 import Mit67, Mit67Sample
from .tinyimagenet import TinyImageNet,TinyImageNetSample


dataset_mean = {
    'tinyimagenet': (0.4802, 0.4481, 0.3975),
}
dataset_std = {
    'tinyimagenet': (0.2770, 0.2691, 0.2822),
}

def get_finegrained_dataloaders(dataset, batch_size=32, num_workers=4, is_instance=False):
    """
    fine grained
    """
    data_folder = os.path.join('../dataset', dataset)
    if dataset in ['dogs', 'cub_200_2011', 'mit67']:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif dataset == 'STL10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])        
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[dataset], dataset_std[dataset])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[dataset], dataset_std[dataset])
        ])

    if dataset == 'cub_200_2011':
        train_set = Cub2011(root=data_folder, train=True, download=False, transform=train_transform)
        test_set = Cub2011(root=data_folder, train=False, download=False, transform=test_transform)
    elif dataset == 'dogs':
        train_set = Dogs(root=data_folder, train=True, download=False, transform=train_transform)
        test_set = Dogs(root=data_folder, train=False, download=False, transform=test_transform) 
    elif dataset == 'mit67':
        train_set = Mit67(root=data_folder, train=True, download=False, transform=train_transform)
        test_set = Mit67(root=data_folder, train=False, download=False, transform=test_transform) 
    elif dataset == 'tinyimagenet':
        train_set = TinyImageNet(root='../dataset', split='train', download=False, transform=train_transform)
        test_set = TinyImageNet(root='../dataset', split='val', download=False, transform=train_transform)
                       


    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)

    return train_loader, test_loader

def get_finegrained_dataloaders_sampler(dataset, batch_size=32, num_workers=4, is_instance=False, k=4096):
    """
    fine grained
    """
    data_folder = os.path.join('/home/zhl/dataset', dataset)
    if dataset in ['dogs', 'cub_200_2011', 'mit67']:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif dataset == 'STL10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])  
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[dataset], dataset_std[dataset])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[dataset], dataset_std[dataset])
        ])

    if dataset == 'cub_200_2011':
        train_set = Cub2011Sample(root=data_folder, train=True, download=False, transform=train_transform, k=k)
        test_set = Cub2011(root=data_folder, train=False, download=False, transform=test_transform)
    elif dataset == 'dogs':
        train_set = DogsSample(root=data_folder, train=True, download=False, transform=train_transform, k=k)
        test_set = Dogs(root=data_folder, train=False, download=False, transform=test_transform) 
    elif dataset == 'mit67':
        train_set = Mit67Sample(root=data_folder, train=True, download=False, transform=train_transform, k=k)
        test_set = Mit67(root=data_folder, train=False, download=False, transform=test_transform) 
    elif dataset == 'tinyimagenet':
        train_set = TinyImageNetSample(root='/home/zhl/dataset', split='train', download=False, transform=train_transform, k=k)
        test_set = TinyImageNet(root='/home/zhl/dataset', split='val', download=False, transform=train_transform)
                       


    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)

    return train_loader, test_loader