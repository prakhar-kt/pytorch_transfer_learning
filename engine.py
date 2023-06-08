import torch
from torch import nn
from tqdm.auto import tqdm

def train_step(model: nn.Module,  
                train_loader: torch.utils.data.DataLoader,
                loss_function: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device):
    
    """
    Performs a single training step for a PyTorch model.
    
    Args: 
        model: a PyTorch model.
        train_loader: a PyTorch DataLoader object.
        loss_function: a PyTorch loss function.
        optimizer: a PyTorch optimizer.
        device: a PyTorch device.

    Returns:
        The loss and accuracy for the training step
    
    """

    model.train()

    loss, acc = 0.0, 0.0

    for X,y in train_loader:    

        X = X.to(device)
        y = y.to(device)

        yhat = model(X)

        loss = loss_function(yhat,y)

        loss += loss.item()

        yhat_probs = torch.softmax(yhat,dim=1)
        yhat_labels = torch.argmax(yhat_probs,dim=1)

        acc += (yhat_labels == y).sum().item() / len(y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    loss /= len(train_loader)
    acc /= len(train_loader)

    return loss, acc

def test_step(model: nn.Module,
              test_loader: torch.utils.data.DataLoader,
              loss_function: nn.Module,
              device: torch.device):
    
    """

    Performs a single testing step for a PyTorch model.

    Args:
        model: a PyTorch model.
        test_loader: a PyTorch DataLoader object.
        loss_function: a PyTorch loss function.
        device: a PyTorch device.

    Returns:
        The loss and accuracy for the testing step.

    """

    loss, acc = 0.0, 0.0
    model.eval()

    with torch.inference_mode():

        for X,y in test_loader:

            X = X.to(device)
            y = y.to(device)

            yhat = model(X)

            loss = loss_function(yhat,y)

            loss += loss.item()

            yhat_probs = torch.softmax(yhat,dim=1)
            yhat_labels = torch.argmax(yhat_probs,dim=1)
            
            acc += (yhat_labels == y).sum().item() / len(y)

    loss /= len(test_loader)
    acc /= len(test_loader)

    return loss, acc

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                loss_function: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                num_epochs: int):
    
    """
    Trains a PyTorch model for a specified number of epochs.

    Args:
        model: a PyTorch model.
        train_loader: a PyTorch DataLoader object.
        test_loader: a PyTorch DataLoader object.
        loss_function: a PyTorch loss function.
        optimizer: a PyTorch optimizer.
        device: a PyTorch device.
        num_epochs: number of epochs to train the model.

    Returns:
        A dictionary containing the losses and accuracies for 
        the training and testing steps.

    """

    results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
                }
    
    for epoch in tqdm(range(num_epochs)):

        train_loss, train_acc = train_step(model,train_loader,loss_function,optimizer,device)
        test_loss, test_acc = test_step(model,test_loader,loss_function,device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: \
              train_loss: {train_loss:.4f}, \
              train_acc: {train_acc:.4f}, \
              test_loss: {test_loss:.4f}, \
              test_acc: {test_acc:.4f}")
        
    return results