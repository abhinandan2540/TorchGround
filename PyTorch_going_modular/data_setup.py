

# # creating dataloaders for model
# def create_dataloader(train_dir: str,
#                       test_dir: str,
#                       transform:
#                       )

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       batch_size: int,
                       transform: transforms.compose,

                       ):

    # making training data
    train_data = datasets.ImageFolder(root=train_dir,  # root of the training data directory
                                      transform=transform,  # transformation of data
                                      # transforms to performs on label (if necessary)
                                      target_transform=None
                                      )

    # making testing data
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform,
                                     target_transform=None
                                     )

    # class name
    class_names = train_data.classes

    # train dataloader
    train_dataloader = DataLoader(dataset=train_data, batch_size=10,  # how many samples per batch
                                  shuffle=True,
                                  # how many subprocess use for data loader? (higher = more)
                                  num_workers=1
                                  )

    # test dataloader
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=1, num_workers=1, shuffle=False)
    return train_dataloader, test_dataloader, class_names
