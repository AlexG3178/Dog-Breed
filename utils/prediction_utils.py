import torch
import matplotlib.pyplot as plt

from utils.data_utils import get_default_device, to_device

device = get_default_device()


def predict_single(img, label, model, breeds):
    xb = to_device(img.unsqueeze(0), device)
    preds = model(xb)
    _, pred = torch.max(preds, dim=1)
    print(f'Actual: {breeds[label]}, Predicted: {breeds[pred.item()]}')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()