import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from unet import UNet

def get_data(
        batch_size=32,
        transform=None,
        target_transform=None,
        num_workers=4,
        pin_memory=True,
):
    test_data = torchvision.datasets.OxfordIIITPet(
        root='segmentation_model/data/test', split='test', target_types='segmentation', download=True, transform=transform, target_transform=target_transform
    )

    train_data = torchvision.datasets.OxfordIIITPet(
        root='segmentation_model/data/train', split='trainval', target_types='segmentation', download=True, transform=transform, target_transform=target_transform
    )
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(train_data, # dataset to turn into iterable
        batch_size=batch_size, # how many samples per batch? 
        shuffle=True, # shuffle data every epoch?
        pin_memory=pin_memory, # speed up data transfer to GPU
        num_workers=num_workers # how many subprocesses to use for data loading
    )

    test_dataloader = DataLoader(test_data,
        batch_size=batch_size, # how many samples per batch? 
        shuffle=False, # don't necessarily have to shuffle the testing data
        pin_memory=pin_memory, # speed up data transfer to GPU
        num_workers=num_workers # how many subprocesses to use for data loading
    )
    return train_dataloader, test_dataloader

def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128), antialias=False),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(contrast=0.3),
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128), antialias=False),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Lambda(tensor_trimap),
])
pets_train_loader, pets_test_loader = get_data(
    batch_size=32,
    transform=transform,
    target_transform=target_transform,
    num_workers=4,
    pin_memory=True,
)
(train_pets_inputs, train_pets_targets) = next(iter(pets_train_loader))
(test_pets_inputs, test_pets_targets) = next(iter(pets_test_loader))

pets_targets_grid = torchvision.utils.make_grid(train_pets_targets.float() / 2.0, nrow=8)