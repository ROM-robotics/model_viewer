"""Shared data model and layer-category heuristics used by all format parsers."""
import re
from dataclasses import dataclass, field


@dataclass
class LayerInfo:
    name: str
    type: str
    category: str = "other"
    extra: str = ""  # optional extra info (shape, param count, etc.)


# Ordered (regex, category) rules. First match wins, so more specific
# patterns (e.g. "layernorm") must come before generic ones (e.g. "norm").
_CATEGORY_RULES = [
    (r"embedding", "embedding"),
    (r"conv", "conv"),
    (r"maxpool|avgpool|adaptive.*pool|pooling|^pool", "pool"),
    (r"batchnorm|layernorm|groupnorm|instancenorm|normalization|^norm$", "norm"),
    (r"lstm|gru|\brnn\b|recurrent", "recurrent"),
    (r"multihead|attention|attn|transformer", "attention"),
    (r"linear|dense|^fc\d*$|matmul|gemm", "linear"),
    (r"dropout", "dropout"),
    (r"relu|gelu|sigmoid|tanh|softmax|silu|swish|elu\b|leakyrelu|activation|mish", "activation"),
    (r"flatten|reshape|^view$|squeeze|unsqueeze|transpose|permute", "reshape"),
    (r"^input$", "input"),
    (r"^output$|identity", "output"),
    (r"add|concat|^sum$|residual", "merge"),
]

CATEGORY_LABELS = {
    "embedding": "Embedding",
    "conv": "Convolution",
    "pool": "Pooling",
    "norm": "Normalization",
    "recurrent": "Recurrent (RNN/LSTM/GRU)",
    "attention": "Attention / Transformer",
    "linear": "Fully Connected (Linear)",
    "dropout": "Dropout",
    "activation": "Activation",
    "reshape": "Reshape",
    "input": "Input",
    "output": "Output",
    "merge": "Merge / Residual",
    "other": "Other",
}

# Mermaid node shape open/close pairs per category, so the diagram gives a
# quick visual hint of what kind of block each node represents.
CATEGORY_SHAPES = {
    "embedding": ("[[", "]]"),
    "conv": ("[", "]"),
    "pool": ("(", ")"),
    "norm": ("{{", "}}"),
    "recurrent": ("([", "])"),
    "attention": ("{{", "}}"),
    "linear": ("[", "]"),
    "dropout": ("(", ")"),
    "activation": ("(", ")"),
    "reshape": ("[/", "/]"),
    "merge": ("((", "))"),
    "other": ("[", "]"),
}

CATEGORY_COLORS = {
    "embedding": "#d4c5f9",
    "conv": "#ffdfba",
    "pool": "#ffe9b3",
    "norm": "#c9e4de",
    "recurrent": "#baffc9",
    "attention": "#d0bfff",
    "linear": "#bae1ff",
    "dropout": "#f0f0f0",
    "activation": "#ffd6e8",
    "reshape": "#e0e0e0",
    "merge": "#ffb3b3",
    "other": "#eeeeee",
}


def categorize(type_name: str) -> str:
    """Map a raw layer/op type name (e.g. 'Conv2d', 'LSTM', 'Relu') to a
    broad architectural category used for grouping and classification."""
    t = type_name.lower()
    for pattern, category in _CATEGORY_RULES:
        if re.search(pattern, t):
            return category
    return "other"


def make_layer(name: str, type_name: str, extra: str = "") -> LayerInfo:
    return LayerInfo(name=name, type=type_name, category=categorize(type_name), extra=extra)


def layers_from_tensor_names(names, shape_lookup=None):
    """Heuristically reconstruct a layer sequence from a flat list of
    parameter/tensor names (used for raw state_dict / safetensors files that
    carry weights only, with no architecture graph).

    Tensor names are grouped by their parent prefix (the name with the final
    '.weight' / '.bias' / etc. component removed) and the prefix's own
    sub-tokens are scanned for well-known layer keywords.
    """
    suffixes = (
        "weight", "bias", "running_mean", "running_var", "num_batches_tracked",
        "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
        "weight_g", "weight_v",
    )
    prefixes = []
    seen = set()
    for n in names:
        prefix = n
        for suf in suffixes:
            if prefix.endswith("." + suf):
                prefix = prefix[: -(len(suf) + 1)]
                break
        if prefix not in seen:
            seen.add(prefix)
            prefixes.append(prefix)

    keyword_to_type = [
        (r"embed", "Embedding"),
        (r"conv", "Conv"),
        (r"\bbn\b|batchnorm|\bnorm\b|ln\b|layernorm", "Norm"),
        (r"lstm", "LSTM"),
        (r"\bgru\b", "GRU"),
        (r"\brnn\b", "RNN"),
        (r"attn|attention", "Attention"),
        (r"\bfc\d*\b|linear|dense|proj|head|classifier", "Linear"),
        (r"pool", "Pool"),
        (r"drop", "Dropout"),
    ]

    layers = []
    for prefix in prefixes:
        short = prefix.rsplit(".", 1)[-1] if "." in prefix else prefix
        guessed = "Unknown"
        for pattern, type_name in keyword_to_type:
            if re.search(pattern, prefix.lower()):
                guessed = type_name
                break
        extra = ""
        if shape_lookup is not None:
            shape = shape_lookup.get(prefix + ".weight") or shape_lookup.get(prefix)
            if shape:
                extra = f"shape={shape}"
        layers.append(make_layer(name=short or prefix, type_name=guessed, extra=extra))
    return layers
