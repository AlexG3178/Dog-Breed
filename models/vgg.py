import torch.nn as nn
from torchvision import models

from models.base import ImageClassificationBase


class DogBreedClassificationVGG(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.vgg16(weights='DEFAULT')
        num_ftrs = self.network.classifier[6].in_features
        self.network.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, xb):
        return self.network(xb)