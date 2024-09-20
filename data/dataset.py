from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import ImageFile

from config import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    

def rename(name):
    new_name = name.split('.')[1]
    new_name = new_name.replace('_',' ')
    return new_name
    
    
def load_data(data_path, batch_size, num_workers):
    dataset = ImageFolder(config['data_path'])
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

    return train_dl, val_dl, test_dl, test_dataset, breeds
