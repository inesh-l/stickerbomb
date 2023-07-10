import torch
import torch.nn as nn
from unet import UNet
from tqdm import tqdm
import torchvision.transforms as transforms
from utils import *

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-3
batch_size = 32
num_epochs = 3
num_workers = 2
image_height = 240
image_width = 240
pin_memory = True
load_model = False
train_dir = "data/oxford-iiit-pet/images"

# Load data
from data import get_data

def train(loader, model, optim, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # backward
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    

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


    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_data, test_data = get_data(
        batch_size=batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(test_data, model, device=device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train(train_data, model, optim, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(test_data, model, device=device)

        # print some examples to a folder
        save_predictions_as_imgs(
            test_data, model, folder="segmentation_model/saved_images/", device=device
        )


if __name__ == '__main__':
    main()

