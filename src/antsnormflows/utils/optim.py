from ..nets.lipschitz import InducedNormLinear, InducedNormConv2d


def set_requires_grad(module, flag):
    """Sets requires_grad flag of all parameters of a torch.nn.module

    Args:
      module: torch.nn.module
      flag: Flag to set requires_grad to
    """

    for param in module.parameters():
        param.requires_grad = flag


def get_requires_grad_states(module):
    """Snapshots the current requires_grad flag of every parameter

    Use together with `restore_requires_grad` to temporarily toggle
    requires_grad (e.g. to disable gradient tracking through part of a
    forward pass) without permanently clobbering parameters that were
    frozen on purpose beforehand.

    Args:
      module: torch.nn.module

    Returns:
      List of booleans, one per parameter (in `module.parameters()` order)
    """
    return [param.requires_grad for param in module.parameters()]


def restore_requires_grad(module, states):
    """Restores requires_grad flags previously captured with
    `get_requires_grad_states`

    Args:
      module: torch.nn.module
      states: List of booleans returned by `get_requires_grad_states`
    """
    for param, state in zip(module.parameters(), states):
        param.requires_grad = state


def clear_grad(model):
    """Set gradients of model parameter to None as this speeds up training,

    See [youtube](https://www.youtube.com/watch?v=9mS1fIYj1So)

    Args:
      model: Model to clear gradients of
    """
    for param in model.parameters():
        param.grad = None


def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, InducedNormConv2d) or isinstance(m, InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)
