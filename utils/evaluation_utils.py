import torch


'''
When you decorate a function with @torch.no_grad(), all the operations performed inside that function will not track gradients
'''
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
