from torch.optim.optimizer import Optimizer, required


class ConventionalSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    This class implements the conventional momentum, like in Lasagne, instead of the one in pytorch
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
    """

    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ConventionalSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ConventionalSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                d_p = -group['lr']*d_p
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p)

        return loss


CSGD = ConventionalSGD

class LSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    This class implements the momentum in Lasagne instead of the one in pytorch
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
    """

    def __init__(self, params, lr=required, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
        super(LSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p = -group['lr']*d_p
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)

                    d_p = buf

                p.data.add_(d_p)

        return loss
