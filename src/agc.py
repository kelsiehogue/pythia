import torch
from torch import nn, optim


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    elif x.ndim == 5:
        dim = [1, 2, 3, 4]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5

class AGC(optim.Optimizer):
    """Generic implementation of the Adaptive Gradient Clipping

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
      optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
      clipping (float, optional): clipping value (default: 1e-3)
      eps (float, optional): eps (default: 1e-3)
      model (torch.nn.Module, optional): The original model
      ignore_agc (str, Iterable, optional): Layers for AGC to ignore
    """

    def __init__(self, params, optim: optim.Optimizer, clipping: float = 1e-2, eps: float = 1e-3, model=None, ignore_agc=["fc"], param_clipping: bool = False):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.optim = optim

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}

        if model is not None:
            assert ignore_agc not in [
                None, []], "You must specify ignore_agc for AGC to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]

            for module_name in ignore_agc:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name))
            params = [{"params": list(module.parameters())} for name,
                          module in model.named_modules() if name not in ignore_agc]
        
        else:
            params = [{"params": params}]

        self.agc_params = params
        self.eps = eps
        self.clipping = clipping
        
        self.param_groups = optim.param_groups
        self.state = optim.state

        self.param_clipping = param_clipping
        #super(AGC, self).__init__([], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.agc_params:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = torch.max(unitwise_norm(
                    p.detach()), torch.tensor(self.eps).to(p.device))
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * self.clipping

                trigger = grad_norm > max_norm

                clipped_grad = p.grad * \
                    (max_norm / torch.max(grad_norm,
                                          torch.tensor(1e-6).to(grad_norm.device)))
                p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))
                
        self.optim.step(closure)

        if self.param_clipping:
            for group in self.agc_params:
                for p in group['params']:
                    p = self.weight_clipping(p)

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This is will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.agc_params:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
    def weight_clipping(self, param):
        param.data = torch.clamp(param.data, -0.001, 0.001)