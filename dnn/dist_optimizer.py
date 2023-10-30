import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required


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


class DistOptimizer(Optimizer):
    r"""Implements stochastic gradient descent for FedAvg."""

    def __init__(self, params, ratio, gmf, mu = 0, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):
        
        self.gmf = gmf
        self.ratio = ratio
        self.etamu = mu * lr
        self.mu = mu

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DistOptimizer, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(DistOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefargs.fracCault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']

                # apply momentum updates
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.etamu != 0:
                    d_p.add_(p.data - param_state['old_init'], alpha=self.mu)

                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)

                else:
                    param_state['cum_grad'].add_(d_p, alpha=local_lr)

                p.data.add_(d_p, alpha=-local_lr)

        return loss

    def average(self, weight):
        param_list = []

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cum_grad'].mul_(weight)
                param_list.append(param_state['cum_grad'])

        communicate(param_list, dist.all_reduce)

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                param_state = self.state[p]

                if self.gmf != 0:
                    if 'global_momentum_buffer' not in param_state:
                        buf = param_state['global_momentum_buffer'] = torch.clone(param_state['cum_grad']).detach()
                        buf.div_(lr)
                    else:
                        buf = param_state['global_momentum_buffer']
                        buf.mul_(self.gmf).add_(param_state['cum_grad'], alpha=1/lr)
                    param_state['old_init'].sub_(lr, buf)
                else:
                    param_state['old_init'].sub_(param_state['cum_grad'])
                
                p.data.copy_(param_state['old_init'])
                param_state['cum_grad'].zero_()

                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()











