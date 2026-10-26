"""Parse PyTorch .pt / .pth files into a LayerInfo sequence.

Handles three shapes of file a user might hand us:
  1. TorchScript archives saved with ``torch.jit.save`` -> introspect modules.
  2. A pickled ``nn.Module`` instance saved with ``torch.save(model, ...)``
     where the original class is importable -> introspect modules.
  3. A plain (or wrapped-in-checkpoint) ``state_dict`` -> no architecture
     graph is available, so layer types are *guessed* from parameter names.
"""
import torch

from .common import make_layer, layers_from_tensor_names


def _layers_from_module(module):
    layers = []
    for name, sub in module.named_modules():
        if name == "":
            continue
        # Only leaf modules (no children) represent actual computation;
        # container modules (Sequential, ModuleList, the model itself) would
        # otherwise duplicate/hide the real layers.
        if next(sub.children(), None) is not None:
            continue
        # TorchScript wraps every submodule in a RecursiveScriptModule proxy;
        # `original_name` recovers the real class name (e.g. "Conv2d") when present.
        type_name = getattr(sub, "original_name", "") or type(sub).__name__
        layers.append(make_layer(name=name, type_name=type_name))
    return layers


def parse(path: str):
    # 1. TorchScript archive
    try:
        scripted = torch.jit.load(path, map_location="cpu")
        layers = _layers_from_module(scripted)
        if layers:
            return layers
    except Exception:
        pass

    # 2. Full pickled object / checkpoint dict
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        layers = _layers_from_module(obj)
        if layers:
            return layers

    state_dict = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state_dict = obj[key]
                break

    if isinstance(state_dict, dict) and state_dict:
        shape_lookup = {}
        names = []
        for k, v in state_dict.items():
            names.append(k)
            if hasattr(v, "shape"):
                shape_lookup[k] = "x".join(str(d) for d in v.shape)
        return layers_from_tensor_names(names, shape_lookup)

    raise ValueError("Unrecognized PyTorch file: no module or state_dict found")
