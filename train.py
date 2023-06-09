import torch
from tqdm import tqdm
import pathlib
from torchvision import transforms
from model import openpose
from loss import PoseLoss
from torch.utils.data import DataLoader
from coco_dataset import CocoKeypoints


# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 1
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False



def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = openpose(in_channels=3)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = PoseLoss()

    train_dataset = CocoKeypoints(
        root=str(pathlib.Path("coco") / "images" / "train2017"),
        annFile=str(
            pathlib.Path("coco")
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
        pin_memory=PIN_MEMORY, # ?
        shuffle=True,
        drop_last=True,
    )
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
