"""Dispatch model files to the right format-specific parser and assemble
the final result used by the web app (layers, architecture label, mermaid)."""
import os

from .architecture import classify_architecture
from .mermaid_gen import generate_mermaid

_EXTENSION_MAP = {
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".onnx": "onnx",
    ".h5": "keras",
    ".keras": "keras",
    ".safetensors": "safetensors",
}

SUPPORTED_EXTENSIONS = sorted(_EXTENSION_MAP.keys())


def _load_parser(kind):
    if kind == "pytorch":
        from . import pytorch_parser as mod
    elif kind == "onnx":
        from . import onnx_parser as mod
    elif kind == "keras":
        from . import keras_parser as mod
    elif kind == "safetensors":
        from . import safetensors_parser as mod
    else:
        raise ValueError(f"Unsupported format: {kind}")
    return mod


def analyze_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    kind = _EXTENSION_MAP.get(ext)
    if kind is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    parser = _load_parser(kind)
    layers = parser.parse(path)

    if not layers:
        raise ValueError("No layers could be extracted from this file")

    primary_label, components = classify_architecture(layers)
    mermaid_source = generate_mermaid(layers)

    return {
        "format": kind,
        "num_layers": len(layers),
        "architecture_primary": primary_label,
        "architecture_components": components,
        "mermaid": mermaid_source,
        "layers": [
            {"name": l.name, "type": l.type, "category": l.category, "extra": l.extra}
            for l in layers
        ],
    }
