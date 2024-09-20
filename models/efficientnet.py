import torch.nn as nn
from torchvision import models

from models.base import ImageClassificationBase


class DogBreedClassificationEfficientNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.efficientnet_b1 (weights='DEFAULT')
        
         # Freeze all layers except the classifier - works faster but less accurate
#         for param in self.network.parameters():
#             param.requires_grad = False
        
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        # Only the parameters of the new classifier will require gradients
        # for param in self.network.classifier.parameters():
        #     param.requires_grad = True
    def forward(self, xb):
        return self.network(xb)