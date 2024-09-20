import torch


config = {
    'n_epochs': 2,
    'batch_size': 32,
    'num_workers': 0,
    'max_learning_rate': 0.001,
    'weight_decay': 1e-3,
    'gradient_clip': 0.1,
    'patience': 5,
    'optimizer': torch.optim.Adam,
    # 'optimizer': torch.optim.SGD,
    'momentum': 0.9,
    'data_path': 'D:/Projects/Dog Breed/datasets/dogimages',
    'model_path': 'D:/Projects/models/dog_breed_model.pth',
}