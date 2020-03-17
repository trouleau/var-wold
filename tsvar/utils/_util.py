import torch


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - x.max())
    return e_x / e_x.sum(dim=0)


def to_numpy(obj):
    if isinstance(obj, list) and isinstance(obj[0], torch.Tensor):
        return [e.numpy() for e in obj]
    elif isinstance(obj, torch.Tensor):
        return e.numpy()
