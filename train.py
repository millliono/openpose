import torch
from tqdm import tqdm
import pathlib
from model import openpose
from loss import PoseLoss
from torch.utils.data import DataLoader
from coco_dataset import CocoKeypoints
from torchvision.transforms import v2
import transforms as mytf
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from coco_eval_model import coco_eval_model
import os

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
BATCH_SIZE = 16
EPOCHS = 200
NUM_WORKERS = 10
MODEL_NAME = "ff0"
LOG_STEP = 1000

def train_fn(train_loader, model, optimizer, loss_fcn, device, epoch, writer):
    model.train()

    loop = tqdm(train_loader, leave=True)
    run_loss = 0.0
    for i, (image, targ_pafs, targ_heatmaps) in enumerate(loop):
        image, targ_pafs, targ_heatmaps = image.to(device), targ_pafs.to(device), targ_heatmaps.to(device)

        pred_pafs, pred_heatmaps = model(image)

        loss = (loss_fcn(pred_pafs, targ_pafs) + loss_fcn(pred_heatmaps, targ_heatmaps)) / BATCH_SIZE
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        run_loss += loss.item()
        if i % LOG_STEP == LOG_STEP - 1:
            avg_loss = run_loss / LOG_STEP
            writer.add_scalar('train_loss', avg_loss, epoch * len(train_loader) + i)
            run_loss = 0.0

        loop.set_postfix(epoch=epoch, loss=loss.item())  # update progress bar
    writer.flush()


def test_fn(test_loader, model, loss_fcn, device, epoch, writer):
    model.eval()

    loop = tqdm(test_loader, leave=True)
    run_vloss = 0.0
    with torch.no_grad():
        for i, (image, targ_pafs, targ_heatmaps) in enumerate(loop):
            image, targ_pafs, targ_heatmaps = image.to(device), targ_pafs.to(device), targ_heatmaps.to(device)

            pred_pafs, pred_heatmaps = model(image)

            vloss = (loss_fcn(pred_pafs, targ_pafs) + loss_fcn(pred_heatmaps, targ_heatmaps)) / BATCH_SIZE
            run_vloss += vloss.item()

    vloss = run_vloss / len(test_loader)
    writer.add_scalar('val_loss', vloss, epoch)
    writer.flush()


def collate_fn(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    pafs = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    heatmaps = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
    return images, pafs, heatmaps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = torch.nn.DataParallel(openpose()).cuda()

        # freeze vgg19 layers
        for param in model.module.backbone.ten_first_layers.parameters():
            param.requires_grad = False
    else:
        model = openpose()

    loss_fcn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    inp_size = 368
    train_dataset = CocoKeypoints(
        root=str(pathlib.Path("../coco") / "images" / "train2017"),
        annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_train2017.json"),
        transform=v2.Compose([mytf.RandomCrop(0.8),
                              mytf.Resize(inp_size),
                              mytf.Pad(inp_size),
                              mytf.RandomRotation(40)]))

    test_dataset = CocoKeypoints(
        root=str(pathlib.Path("../coco") / "images" / "val2017"),
        annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
        transform=v2.Compose([mytf.RandomCrop(0.8), mytf.Resize(inp_size),
                              mytf.Pad(inp_size)]))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,  
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,  
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    dir = os.path.join("runs", MODEL_NAME)
    writer = SummaryWriter(dir)

    best_mAP = float('inf')
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fcn, device, epoch, writer)
        test_fn(test_loader, model, loss_fcn, device, epoch, writer)

        mAP = coco_eval_model(model, device)
        writer.add_scalar('mAP', mAP, epoch)
        writer.flush()

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), MODEL_NAME + ".pth")
            print(f"best model at epoch: {epoch}")
    writer.close()


if __name__ == "__main__":
    main()
