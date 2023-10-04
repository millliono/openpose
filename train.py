import torch
from tqdm import tqdm
import pathlib
from torchvision import transforms
from model import openpose
from loss import PoseLoss
from torch.utils.data import DataLoader
from coco_dataset import CocoKeypoints

# Hyperparameters etc.
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
WEIGHT_DECAY = 0
EPOCHS = 1
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(train_loader, model, optimizer, loss_fn, device):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (image, targ_pafs, targ_heatmaps, mask_out) in enumerate(loop):
        image = image.to(device)
        targ_pafs = targ_pafs.to(device)
        targ_heatmaps = targ_heatmaps.to(device)
        mask_out = mask_out.to(device)

        pred_pafs, pred_htmps, save_for_loss_pafs, save_for_loss_htmps = model(image)

        loss = loss_fn(
            save_for_loss_pafs, save_for_loss_htmps, targ_pafs, targ_heatmaps, mask_out
        )
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
    mask_out = torch.utils.data.dataloader.default_collate([b[3] for b in batch])
    return images, pafs, heatmaps, mask_out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # comment when using modern gpu

    model = openpose().to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    model.train()

    # freeze vgg19 layers
    for param in model.module.backbone.ten_first_layers.parameters():
        param.requires_grad = False

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
        input_transform=transforms.Compose(
            [
                transforms.Resize((368, 368)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        heatmaps_transform=transforms.Resize(
            (46, 46), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False
        ),
        pafs_transform=transforms.Resize(
            (46, 46), interpolation=transforms.InterpolationMode.NEAREST, antialias=False
        ),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,  # ?
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, device)

    torch.save(model.state_dict(), "save_model.pth")
    print("Saved openpose to save_model.pth")


if __name__ == "__main__":
    main()
