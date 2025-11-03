"""
Handles loading and prepping the data from Caltech-101
"""

import os
import torch 
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset


def get_transformation():

    data_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return data_transform

def load_dataset():
    
    download_flag = True

    dataset_path = "./data/caltech101/101_ObjectCategories"
    if os.path.exists(dataset_path):
        print("Dataset found locally.")
        download_flag = False
    else:
        print("Downloading Caltech 101 dataset.")

    caltech_dataset = datasets.Caltech101(
        root = "./data/",
        transform = get_transformation(),
        download= download_flag
    )
    print(f"Loaded dataset.\n\nDataset Size: {len(caltech_dataset)}\n\nCategories:{caltech_dataset.categories}")

    return caltech_dataset

class SafeDataset(Dataset):
    """Wraps a dataset to skip any samples that cause errors (e.g., corrupt images)."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, idx):
        try:
            return self.base_dataset[idx]
        except Exception as e:
            print(f"Skipping sample {idx}: {e}")
            # move to the next sample safely
            return self.__getitem__((idx + 1) % len(self.base_dataset))

    def __len__(self):
        return len(self.base_dataset)

def split_dataset(dataset, train_fraction=0.75, seed=42):
    print("Splitting the dataset.")
    
    n = len(dataset)
    n_train = int(n * train_fraction)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    return train_set, val_set

# def make_dataloaders(train_set, val_set, batch_size=32, num_workers=2):
#     print("Loading the data.")
    

    
#     return train_loader, val_loader




def get_data_loaders():

    caltech_dataset = load_dataset()
    caltech_dataset = SafeDataset(caltech_dataset)

    train_set, val_set = split_dataset(caltech_dataset)

    train_loader = DataLoader(train_set, 
                              batch_size=32, 
                              shuffle=True, 
                              num_workers=0)
    val_loader = DataLoader(val_set, 
                            batch_size=32, 
                            shuffle=False, 
                            num_workers=0)

    return train_loader, val_loader, caltech_dataset


t1, v1, ds = get_data_loaders()