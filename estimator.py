import torch
from PIL import Image
from torchvision import transforms
from urllib import request
import matplotlib.pyplot as plt

import model


# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    request.urlretrieve(url, filename)
except Exception as e:
    print(f"An error occurred while downloading the file: {e}")


input_image = Image.open(filename)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
print(input_batch.shape)

cnn = model.openpose(in_channels=3)
cnn.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    cnn.to("cuda")
with torch.no_grad():
    output_batch = cnn(input_batch)
print(output_batch.shape)

for output in output_batch:
    part_aff = output[:52]
    print(part_aff.shape)
    conf_map = output[52:]
    print(conf_map.shape)

    # i choose to continue with 28x28 size images and not the original size
    # but i think in pytorch implemetaton it resizes to original


