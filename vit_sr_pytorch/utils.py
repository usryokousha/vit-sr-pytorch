import torch
import torch.nn.functional as F


def create_correlation_mask(
    batch: int,
    shape: tuple, 
    tx: tuple = None, 
    ty: tuple = None, 
    device: torch.device = None):
    """Build mask to restrict correlation to a given area."""
    if tx is None:
        tx = (torch.tensor((0.,), device=device),
                torch.tensor((1.,), device=device))
    if ty is None:
        ty = (torch.tensor((0.,), device=device),
                torch.tensor((1.,), device=device))
    
    mask = torch.ones((batch,) + shape, device=device, dtype=torch.float)
    for dim, (pos, sigma) in enumerate((ty, tx)):
        length = shape[dim] // 2
        domain = torch.linspace(-length, -length + shape[dim], shape[dim], 
        device=device, dtype=torch.float)

        vals = torch.exp(-(domain[None, :] - pos[:, None]) ** 2 / sigma[:, None] ** 2)

        if dim == 0:
            mask *= vals[:, :, None]    
        else:
            mask *= vals[:, None, :]

    return mask
