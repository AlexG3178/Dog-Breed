import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

dataset = ImageFolder('D:/Projects/Dog Breed/Data/dogimages')
breeds = []
print('CUDA Available =', torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

n_epochs = 15
max_leaning_rate = 0.01
weight_decay = 1e-4
momentum = 0.9
gradient_clip = 0.1
batch_size = 32
num_workers = 0
# optimizer = torch.optim.Adam
optimizer = torch.optim.SGD


#region Dataset & Transforms
def rename(name):
    new_name = name.split('.')[1]
    new_name = new_name.replace('_',' ')
    return new_name

for name in dataset.classes:
    breeds.append(rename(name))

train_ds, test_dev_ds = train_test_split(dataset, test_size=0.4, train_size=0.6, random_state=34)
val_ds, test_ds = train_test_split(test_dev_ds, test_size=0.5, train_size=0.5, random_state=34)
print(len(train_ds), len(val_ds), len(test_ds))


class DogsDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]

        # Debugging: Check the type of the image before and after transformation
        # print(f"Before transform: {type(img)}")

        if self.transform:
            try:
                img = self.transform(img)
                # print(f"After transform: {type(img)}")
            except Exception as e:
                print(f"Error during transformation: {e}")
                raise

        return img, label
        
        
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),  # Random crop
    transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flip
    transforms.RandomRotation(degrees=30),  # Random rotation
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

train_dataset = DogsDataset(train_ds, train_transform)
val_dataset = DogsDataset(val_ds, val_transform)
test_dataset = DogsDataset(test_ds, test_transform)
        

train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size*2, num_workers=num_workers, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size*2, num_workers=num_workers, pin_memory=True)

#endregion



#region: Model

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch+1}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        


class DogBreedClassificationCNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            # Convolutional Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces 224x224 -> 112x112
            
            # Convolutional Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces 112x112 -> 56x56

            # Convolutional Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces 56x56 -> 28x28

            # Convolutional Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces 28x28 -> 14x14

            # Flatten Layer
            nn.Flatten(),

            # Fully Connected Layer 1
            nn.Dropout(0.5),
            nn.Linear(256*14*14, 512),
            nn.ReLU(),

            # Fully Connected Layer 2 (Output Layer)
            nn.Dropout(0.5),
            nn.Linear(512, len(breeds)),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DogBreedClassificationEfficientNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(breeds)),
            nn.Softmax(dim=1)
        )
    def forward(self, xb):
        return self.network(xb)
    

class DogBreedClassificationResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(weights='DEFAULT')
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(breeds)),
            nn.Softmax(dim=1)
        )
    def forward(self, xb):
        return self.network(xb)


class DogBreedClassificationVGG(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.vgg16(weights='DEFAULT')
        num_ftrs = self.network.classifier[6].in_features
        self.network.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(breeds)),
            nn.Softmax(dim=1)
        )
    def forward(self, xb):
        return self.network(xb)
    
#endregion

# model = DogBreedClassificationCNN()
# Epoch [10], train_loss: 4.8556, val_loss: 4.8793, val_acc: 0.0087

model = DogBreedClassificationEfficientNet()
# Epoch [10], train_loss: 0.5459, val_loss: 0.8211, val_acc: 0.7502

# model = DogBreedClassificationResNet()
# Epoch [10], train_loss: 1.0750, val_loss: 1.0798, val_acc: 0.6796

# model = DogBreedClassificationVGG()
# Epoch [10], train_loss: 4.8558, val_loss: 4.8786, val_acc: 0.0104


#region Device Configuration
    
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    else:
        return data.to(device, non_blocking=True)
    

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)
            torch.cuda.empty_cache()

#endregion



#region Training
                                              
'''
When you decorate a function with @torch.no_grad(), all the operations performed inside that function will not track gradients
'''
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, max_lr=0.01, weight_dec = 1e-4, moment=0.0, grad_clip=None, opt=optimizer):
    
    torch.cuda.empty_cache()
    history = []
    optim = opt(model.parameters(), max_lr, weight_decay=weight_dec, momentum=moment)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optim.step()
            optim.zero_grad()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history

#endregion



#region  Main function

if __name__ == "__main__":
    
    # Device configuration
    device = get_default_device()
    print(f'default device: {device}')
    
    # Moving data and model to GPU
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)
    
    # Training
    history = fit(n_epochs, model, train_dl, val_dl, max_leaning_rate, weight_dec=weight_decay, moment=momentum, grad_clip=gradient_clip)
    
    # Plotting the training history
    epochs = range(len(history))
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    val_accs = [x['val_acc'] for x in history]
    
    # Evaluate on test data
    test_dl = DeviceDataLoader(test_dl, device)
    result = evaluate(model, test_dl)
    print(f"Test Loss: {result['val_loss']:.4f}, Test Accuracy: {result['val_acc']:.4f}")
    
    
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, '-o', label='Train loss')
    plt.plot(epochs, val_losses, '-o', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, '-o', label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

    
    
    # Prediction
    # def predict_single(img, label):
    #     xb = to_device(img.unsqueeze(0), device)
    #     preds = model(xb)
    #     _, pred = torch.max(preds, dim=1)
    #     print(f'Actual: {breeds[label]}, Predicted: {breeds[pred.item()]}')
    #     plt.imshow(img.permute(1, 2, 0))
    #     plt.show()

    # # Test a prediction
    # predict_single(*test_dataset[8])
    
    #endregion