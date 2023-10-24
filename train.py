import torch
from tqdm import tqdm
import pathlib
from torchvision import transforms
from model import openpose
from loss import PoseLoss
from torch.utils.data import DataLoader
from coco_dataset import CocoKeypoints
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters etc.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 1
EPOCHS = 1
NUM_WORKERS = 4
COMMENT = "LR_1e-4_BATCH_16"
LOG_STEP = 100
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(train_loader, model, optimizer, loss_fn, device, epoch, writer):
    loop = tqdm(train_loader, leave=True)

    run_loss = 0.0
    for i, (image, targ_pafs, targ_heatmaps) in enumerate(loop):
        image = image.to(device)
        targ_pafs = targ_pafs.to(device)
        targ_heatmaps = targ_heatmaps.to(device)

        _, _, save_for_loss_pafs, save_for_loss_htmps = model(image)

        loss = loss_fn(save_for_loss_pafs, save_for_loss_htmps, targ_pafs, targ_heatmaps)

        run_loss += loss.item()
        if i % LOG_STEP == LOG_STEP - 1:
            writer.add_scalar('train_loss', run_loss / LOG_STEP, epoch * len(train_loader) + i)
            run_loss = 0.0

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())  # update progress bar
    writer.flush()


def collate_fn(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    pafs = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    heatmaps = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
    return images, pafs, heatmaps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # comment if using modern gpu

    if device == "cuda":
        model = torch.nn.DataParallel(openpose()).cuda()

        # freeze vgg19 layers
        for param in model.module.backbone.ten_first_layers.parameters():
            param.requires_grad = False
    else:
        model = openpose()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = PoseLoss()

    inp_size = (368, 368)
    targ_size = (46, 46)
    train_dataset = CocoKeypoints(
        root=str(pathlib.Path("../coco") / "images" / "train2017"),
        annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_train2017.json"),
        input_transform=transforms.Compose([
            transforms.Resize(inp_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        targ_size=targ_size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,  # ?
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    writer = SummaryWriter(comment=COMMENT)
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, device, epoch, writer)
    writer.close()

    torch.save(model.state_dict(), "save_model.pth")
    print("Saved openpose to save_model.pth")


if __name__ == "__main__":
    main()
