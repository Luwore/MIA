import torch
from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_resnet18(num_classes=10):
    model = resnet18(weights=None, num_classes=num_classes).to(device)
    return model


