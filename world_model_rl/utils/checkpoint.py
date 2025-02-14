import torch
import os

def save_checkpoint(model, optimizer, epoch, filepath="checkpoints/model.pth"):
    """
    Saves the model checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(model, optimizer, filepath="checkpoints/model.pth"):
    """
    Loads the model checkpoint.
    """
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch']
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0
