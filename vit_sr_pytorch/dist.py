import torch.distributed as dist

# distributed helper
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
  
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if get_world_size() == 1:
        return tensor
    
    dist.all_reduce(tensor, op=op)
    return tensor