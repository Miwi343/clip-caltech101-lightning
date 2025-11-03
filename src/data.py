"""
Handles loading and prepping the data from Caltech-101
"""

import torch 
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


def get_transformation():

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return data_transform

def load_dataset():
    
    print("Loading Caltech 101 dataset.")

    caltech_dataset = datasets.Caltech101(
        root = "./data/",
        transform = get_transformation(),
        download= True
    )
    print(f"Downloaded dataset.\n\nDataset Size: {len(caltech_dataset)}\n\nCategories:{caltech_dataset.categories}")

    return caltech_dataset

def split_dataset(dataset, train_fraction=0.75, seed=42):
    print("Splitting the dataset.")
    
    n = len(dataset)
    n_train = int(n * train_fraction)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    return train_set, val_set

def make_dataloaders(train_set, val_set, batch_size=32, num_workers=2):
    print("Loading the data.")
    
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)
    
    return train_loader, val_loader




def get_data_loaders():

    caltech_dataset = load_dataset()

    train_set, val_set = split_dataset(caltech_dataset)

    return make_dataloaders(train_set, val_set)


t1, v1 = get_data_loaders()