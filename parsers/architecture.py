"""Classify a parsed layer sequence into a human-friendly architecture
label (CNN / RNN / Gated RNN / Transformer / MLP / hybrid combinations)."""


def classify_architecture(layers):
    categories = {l.category for l in layers}
    types_lower = [l.type.lower() for l in layers]

    components = []

    if "conv" in categories:
        components.append("CNN (Convolutional Neural Network)")

    has_lstm = any("lstm" in t for t in types_lower)
    has_gru = any("gru" in t for t in types_lower)
    has_plain_rnn = any(t in ("rnn", "rnncell", "recurrent") for t in types_lower)
    if has_lstm or has_gru:
        gate_kind = "/".join([n for n, present in (("LSTM", has_lstm), ("GRU", has_gru)) if present])
        components.append(f"Gated RNN ({gate_kind})")
    elif has_plain_rnn or "recurrent" in categories:
        components.append("RNN (Recurrent Neural Network)")

    if "attention" in categories or any("transformer" in t for t in types_lower):
        components.append("Transformer (Attention-based)")

    if "embedding" in categories and not components:
        components.append("Embedding-based model")

    if not components:
        if "linear" in categories:
            components.append("Feedforward Neural Network / MLP")
        else:
            components.append("Unknown / Custom architecture")

    if len(components) > 1:
        primary = "Hybrid: " + " + ".join(c.split(" (")[0] for c in components)
    else:
        primary = components[0]

    return primary, components
