import torch
import torch.nn as nn
from tqdm import tqdm

from utils.evaluation_utils import evaluate


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