import torch

def flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def communicate(tensors, communication_op, attention=False):
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    if attention:
        return tensors/flat_tensor
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        t.set_(f)
