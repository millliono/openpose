import torch
from tqdm import tqdm
import os
import pathlib
import model as mdl
import torch.nn as nn
from torch.utils.data import DataLoader
import transforms as mytf
from loss import PoseLoss
from coco_dataset import CocoKeypoints
from torchvision.transforms import v2
from coco_eval_model import coco_eval_model
from torch.utils.tensorboard import SummaryWriter

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0
BATCH_SIZE = 10
EPOCHS = 400
NUM_WORKERS = 10
MODEL_NAME = "fff#"
LOG_STEP = 1000


def train_fn(dataloader, model, optimizer, loss_fcn, device, epoch, writer):
    model.train()

    loop = tqdm(dataloader, leave=True)
    run_loss = 0.0
    for i, (image, targ_pafs, targ_heatmaps) in enumerate(loop):
        image, targ_pafs, targ_heatmaps = image.to(device), targ_pafs.to(device), targ_heatmaps.to(device)

        _, _, save_for_loss_pafs, save_for_loss_htmps = model(image)

        loss = loss_fcn(save_for_loss_pafs, save_for_loss_htmps, targ_pafs, targ_heatmaps)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        run_loss += loss.item()
        if i % LOG_STEP == LOG_STEP - 1:
            writer.add_scalar('train_loss', run_loss / LOG_STEP, epoch * len(dataloader) + i)
            run_loss = 0.0

        loop.set_postfix(epoch=epoch, loss=loss.item())
    writer.flush()


@torch.no_grad()
def test_fn(dataloader, model, loss_fcn, device, epoch, writer):
    model.eval()

    loop = tqdm(dataloader, leave=True)
    run_vloss = 0.0
    for i, (image, targ_pafs, targ_heatmaps) in enumerate(loop):
        image, targ_pafs, targ_heatmaps = image.to(device), targ_pafs.to(device), targ_heatmaps.to(device)

        _, _, save_for_loss_pafs, save_for_loss_htmps = model(image)

        vloss = loss_fcn(save_for_loss_pafs, save_for_loss_htmps, targ_pafs, targ_heatmaps)
        run_vloss += vloss.item()

    vloss = run_vloss / len(dataloader)
    writer.add_scalar('val_loss', vloss, epoch)
    writer.flush()


def collate_fn(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    pafs = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    heatmaps = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
    return images, pafs, heatmaps


def main():
    writer = SummaryWriter(os.path.join("runs", MODEL_NAME))

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(mdl.openpose(), device_ids=[1, 2, 3]).to(device)

    loss_fcn = PoseLoss(BATCH_SIZE, reduction='sum')

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

    mAP_dataset = CocoKeypoints(
        root=str(pathlib.Path("../coco") / "images" / "val2017"),
        annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
        transform=None)

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

    # freeze vgg19 layers
    for param in model.module.backbone.ten_first_layers.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    best_mAP = 0
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fcn, device, epoch, writer)
        test_fn(test_loader, model, loss_fcn, device, epoch, writer)

        if epoch >= 20:
            mAP = coco_eval_model(mAP_dataset, model, device)
            writer.add_scalar('mAP', mAP, epoch)
            writer.flush()

            if mAP > best_mAP:
                torch.save(model.state_dict(), MODEL_NAME + ".pth")
                best_mAP = mAP
    writer.close()


if __name__ == "__main__":
    main()
