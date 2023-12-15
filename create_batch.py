import torch
from torch.utils.data import DataLoader

def collate_batch(batch):
    """
    Returns the input and target tensors for a batch of data

    Parameters:
        `batch` - An iterable data structure of tuples (img, label),
                  where `img` is an image, and
                  `label` is either 1 or 0. for yes or no classification 
                  of melanoma for the corresponding image.

    Returns: a tuple `(X, t)`, where
        - `img_list` is a PyTorch tensor of shape (batch_size,  img size)
        - `target_list` is a PyTorch tensor of shape (batch_size)
    """
    img_list = []  # collect each sample's sequence of word indices
    target_list = [] # collect each sample's target labels
    for (img, label) in batch:
        img_list.append(torch.tensor(img))
        target_list.append(torch.tensor(label))

    return img_list, target_list
