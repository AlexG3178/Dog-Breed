import torch.nn as nn

from models.base import ImageClassificationBase


class DogBreedClassificationCNN(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces size by half
            
            # Convolutional Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces size by half
            
            # Convolutional Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces size by half
            
            # Flatten Layer
            nn.Flatten(),
            
            # Fully Connected Layer 1
            nn.Dropout(0.4),
            nn.Linear(128*28*28, 512),
            nn.ReLU(),
            
            # Fully Connected Layer 2 (Output Layer)
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)