import torch
from tqdm import tqdm
import pathlib
from torchvision import transforms
from model import openpose
from loss import PoseLoss
from torch.utils.data import DataLoader
from coco_dataset import CocoKeypoints
import torchvision.transforms.functional as fcn
import numpy as np

# Hyperparameters etc.
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1
NUM_WORKERS = 16
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (image, pafs, heatmaps) in enumerate(loop):
        image = image.to(DEVICE)
        pafs = pafs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)
        out, save_for_loss_pafs, save_for_loss_htmps = model(image)
        loss = loss_fn(save_for_loss_pafs, save_for_loss_htmps, pafs, heatmaps)
        print(f"Batch-({batch_idx}) loss was {loss}")

        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def collate_fn(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    pafs = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    heatmaps = torch.utils.data.dataloader.default_collate([b[2] for b in batch])

    return images, pafs, heatmaps


def main():
    model = openpose(in_channels=3).to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = PoseLoss()

    train_dataset = CocoKeypoints(
        root=str(pathlib.Path("../coco") / "images" / "train2017"),
        annFile=str(
            pathlib.Path("../coco")
            / "annotations"
            / "annotations"
            / "person_keypoints_train2017.json"
        ),
        transform=transforms.Resize([224, 224]),
        resize_keypoints_to=[224, 224],
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,  # ?
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

    torch.save(model.state_dict(), "save_model.pth")
    print("Saved openpose to save_model.pth")


if __name__ == "__main__":
    main()
