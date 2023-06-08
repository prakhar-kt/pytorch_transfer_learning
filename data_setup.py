import os
from torchvision import datasets,transforms

from torch.utils.data import DataLoader

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int = 32,
):
    """
    Creates PyTorch DataLoader objects ready for training and testing a model.

    Args:   
        train_dir: directory of training images.
        test_dir: directory of test images.
        transform: a composition of image transformations.
        batch_size: number of images to load per batch.
    
    Returns:
        A tuple of (train_data, test_data) where each is a PyTorch DataLoader object.
        
    """
    train_data = datasets.ImageFolder(root=train_dir,transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,transform=transform)

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

    class_names = train_data.classes

    return train_loader,test_loader,class_names

