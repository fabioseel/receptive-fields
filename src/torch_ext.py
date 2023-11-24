import torch

class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)

class ActivationRegularization():
    def __init__(self, module = torch.nn.Module | list, p=2.0, act_lambda=0.01):
        self._lambda = act_lambda
        if act_lambda > 0:
            self.hook = OutputHook()
            self.p = p
            if isinstance(module, torch.nn.Module):
                module.register_forward_hook(self.hook)
            else:
                for m in module:
                    m.register_forward_hook(self.hook)

    def penalty(self):
        penalty = 0.
        if self._lambda > 0:
            for output in self.hook:
                penalty += torch.norm(output, self.p)
            penalty *= self._lambda
            self.hook.clear()
        return penalty