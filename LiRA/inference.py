import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 128
save_path = 'exp/cifar10/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def perform_inference(model, dataloader):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            logits_list.append(outputs.cpu().numpy())
    return np.concatenate(logits_list)


def main():
    for i in range(16):
        model = resnet18(weights=None, num_classes=10).to(device)
        model_dir = os.path.join(save_path, f'experiment_{i + 1}_of_16')
        model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt.pth')))

        logits = perform_inference(model, dataloader)
        logits_dir = os.path.join(model_dir, 'logits')
        os.makedirs(logits_dir, exist_ok=True)
        np.save(os.path.join(logits_dir, 'logits.npy'), logits)


if __name__ == '__main__':
    main()
