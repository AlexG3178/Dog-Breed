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



#region Load data & Transforms

def load_data(data_path, batch_size, num_workers):
    dataset = ImageFolder(data_path)
    breeds = [rename(name) for name in dataset.classes]

    train_ds, test_dev_ds = train_test_split(dataset, test_size=0.4, train_size=0.6, random_state=34)
    val_ds, test_ds = train_test_split(test_dev_ds, test_size=0.5, train_size=0.5, random_state=34)
    print(f"train_ds: {len(train_ds)}, val_ds: {len(val_ds)}, test_ds: {len(test_ds)}") 
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image
        transforms.RandomCrop(224, padding=4, padding_mode='reflect'),  # Random crop
        transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flip
        transforms.RandomRotation(degrees=30),  # Random rotation
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])  # Normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])  # Normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])  # Normalize
    ])

    train_dataset = DogsDataset(train_ds, train_transform)
    val_dataset = DogsDataset(val_ds, val_transform)
    test_dataset = DogsDataset(test_ds, test_transform)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size * 2, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size * 2, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl, breeds


def rename(name):
    new_name = name.split('.')[1]
    new_name = new_name.replace('_',' ')
    return new_name


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

#endregion



#region: Model
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


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))        


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
    
#
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
    
#endregion



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

def fit(epochs, model, train_loader, val_loader, max_lr, weight_dec, momentum, grad_clip, optimizer, patience):
    torch.cuda.empty_cache()
    history = []
    
    # Adam
    optimizer = optimizer(model.parameters(), max_lr, weight_decay=weight_dec)
    # SGD
    # optimizer = optimizer(model.parameters(), max_lr, weight_decay=weight_dec, momentum=momentum)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    '''Best for: Training CNNs from scratch on large datasets.
    It adjusts the learning rate according to the One Cycle Policy, which increases 
    the learning rate to a maximum value and then decreases it towards the end of training. 
    This can often lead to faster convergence and better generalization.
    '''
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=patien)
    ''' Best for: Fine-tuning pre-trained models or when training CNNs where you want 
    to reduce the learning rate based on the performance (e.g., validation loss).
    It monitors a metric (usually validation loss) and reduces the learning rate when the metric stops improving, 
    which can help in escaping local minima and achieving better convergence.
    '''
    best_val_loss = float('inf')
    epochs_without_improvement = 0

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

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            # scheduler.step(metrics=loss.item()) # for ReduceLROnPlateau
            lrs.append(scheduler.get_last_lr()[0])

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs

        model.epoch_end(epoch, result)
        history.append(result)

        # Early Stopping Check
        val_loss = result['val_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset the counter if validation loss improves
        else:
            epochs_without_improvement += 1  # Increment the counter if validation loss does not improve

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break  # Exit the training loop

    return history

#endregion



#region  Main function

def main():
    
    # Configuration
    data_path = 'D:/Projects/Dog Breed/Data/dogimages'
    model_path = 'D:/Projects/models/dog_breed_model.pth'
    batch_size = 32
    num_workers = 0
    n_epochs = 10
    max_learning_rate = 0.001
    weight_decay = 1e-3
    gradient_clip = 0.1
    patience = 5
    optimizer = torch.optim.Adam
    # optimizer = torch.optim.SGD
    momentum = 0.9
    
    print('CUDA Available =', torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    
     # Load data
    train_dl, val_dl, test_dl, breeds = load_data(data_path, batch_size, num_workers)

    # Initialize model
    num_classes = len(breeds)
    # model = DogBreedClassificationCNN(num_classes)
    model = DogBreedClassificationEfficientNet(num_classes)
    # model = DogBreedClassificationResNet(num_classes)
    # model = DogBreedClassificationVGG(num_classes)

    # Device configuration
    device = get_default_device()
    print(f'default device: {device}')
    
    # Moving data and model to GPU
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)
    to_device(model, device)
    
    # Training
    history = fit(n_epochs, model, train_dl, val_dl, max_learning_rate, weight_decay, momentum, gradient_clip, optimizer, patience)
    
     # Evaluate on test data
    test_dl = DeviceDataLoader(test_dl, device)
    result = evaluate(model, test_dl)
    print(f"Test Loss: {result['val_loss']:.4f}, Test Accuracy: {result['val_acc']:.4f}")
    
    # Save the model=
    torch.save(model, model_path)
    
    # Plotting the training history
    epochs = range(len(history))
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    val_accs = [x['val_acc'] for x in history]

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
    
    
if __name__ == '__main__': 
    main()
    
    #endregion
   
