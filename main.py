import torch
import matplotlib.pyplot as plt

from models.cnn import DogBreedClassificationCNN
from models.efficientnet import DogBreedClassificationEfficientNet
from models.resnet import DogBreedClassificationResNet
from models.vgg import DogBreedClassificationVGG
from data.dataset import load_data
from utils.data_utils import DeviceDataLoader, get_default_device, to_device
from utils.training_utils import fit
from utils.prediction_utils import predict_single
from utils.evaluation_utils import evaluate
from config import config


def main():
 
    print('CUDA Available =', torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    
    train_dl, val_dl, test_dl, test_dataset, breeds = load_data(config['data_path'], config['batch_size'], config['num_workers'])

    num_classes = len(breeds)
    # model = DogBreedClassificationCNN(num_classes)
    model = DogBreedClassificationEfficientNet(num_classes)
    # model = DogBreedClassificationResNet(num_classes)
    # model = DogBreedClassificationVGG(num_classes)

    device = get_default_device()
    print(f'default device: {device}')
    
    # Moving data and model to GPU
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)
    to_device(model, device)
    
    # Training
    history = fit(config['n_epochs'], model, train_dl, val_dl, config['max_learning_rate'], config['weight_decay'], 
                  config['momentum'], config['gradient_clip'], config['optimizer'], config['patience'])
    
    # Evaluate on test data
    test_dl = DeviceDataLoader(test_dl, device)
    result = evaluate(model, test_dl)
    print(f"Test Loss: {result['val_loss']:.4f}, Test Accuracy: {result['val_acc']:.4f}")
    
    torch.save(model, config['model_path'])
    
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

    predict_single(*test_dataset[8], model, breeds)
    
    
if __name__ == '__main__': 
    main()
    

   
