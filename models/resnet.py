import torch.nn as nn
from torchvision import models

from models.base import ImageClassificationBase



class DogBreedClassificationResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(weights='DEFAULT')
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, xb):
        return self.network(xb)