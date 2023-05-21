import torch
import model

# Create an instance of the PyTorch model
model = model.openpose(in_channels=3)

# Provide dummy input tensor to the model
dummy_input = torch.randn(1, 3, 224, 224)

# Export the PyTorch model to ONNX format
torch.onnx.export(model, dummy_input, 'openpose.onnx')