Overview
This repository provides an end-to-end pipeline for training and evaluating dog breed classification models using deep learning. 
The project includes:

1. Dataset Management: Loading and processing images from a dataset of dog breeds, including train/validation/test splits and data augmentation techniques.
2. Custom CNN Model: Implementation of a custom convolutional neural network for image classification.
3. Pre-Trained Models: Utilization of popular pre-trained models such as EfficientNet, ResNet, and VGG for transfer learning and fine-tuning.
4. Training Pipeline: The training loop supports dynamic learning rate scheduling, gradient clipping, and early stopping for better generalization.
5. Evaluation and Prediction: Tools for evaluating model performance and making predictions on single images.
6. Device Management: Automatic GPU/CPU selection and efficient data loading with PyTorch DataLoader.

Features
• Custom and Pre-Trained Models: Train a custom CNN or use pre-trained models like EfficientNet, ResNet, and VGG.
• Data Augmentation: Random cropping, flipping, rotation, and normalization to improve model robustness.
• Early Stopping: Prevent overfitting by stopping training if validation loss stops improving.
• Learning Rate Scheduling: Adaptive learning rates for better convergence using OneCycleLR or ReduceLROnPlateau.
• Automatic Device Selection: The code automatically selects GPU if available, otherwise, it defaults to CPU.
• Inference: Predict dog breeds from a single image with model inference tools.

Requirements
To run the project, the following libraries are required:
• PyTorch
• Torchvision
• scikit-learn
• Matplotlib
• tqdm
• PIL (Pillow)
