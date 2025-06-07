import torch
from ultralytics import YOLO
import pnnx

model = YOLO("yolo11n.pt")

x = torch.rand(1, 3, 640, 640)

opt_model = pnnx.export(model, "yolo11n.pt", x)

# use tuple for model with multiple inputs
# opt_model = pnnx.export(model, "resnet18.pt", (x, y, z))
