import torch
def accuracy(model, data, device="cuda:0"):
    """
    Accuracy function, self explanatory.
    """
    loader = torch.utils.data.DataLoader(data, batch_size=32)
    model.to(device)
    model.eval() # annotate model for evaluation (important for batch normalization)
    correct = 0
    total = 0
    for imgs, labels in loader:
        labels = labels.to(device)
        output = model(imgs.to(device))
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

