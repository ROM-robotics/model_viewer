"""Parse ONNX .onnx files into a LayerInfo sequence using the onnx graph."""
import onnx

from .common import make_layer


def _shape_str(value_info):
    try:
        dims = value_info.type.tensor_type.shape.dim
        parts = []
        for d in dims:
            if d.dim_value:
                parts.append(str(d.dim_value))
            elif d.dim_param:
                parts.append(d.dim_param)
            else:
                parts.append("?")
        return "x".join(parts)
    except Exception:
        return ""


def parse(path: str):
    # We only need the graph structure (op types, input/output names), not
    # the actual tensor values, so skip loading external weight data files
    # (e.g. "model.onnx.data") that may not be present alongside the upload.
    model = onnx.load(path, load_external_data=False)
    graph = model.graph

    initializer_names = {init.name for init in graph.initializer}

    layers = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue  # it's a weight tensor exposed as a graph input, not real input
        layers.append(make_layer(name=inp.name, type_name="Input", extra=_shape_str(inp)))

    for node in graph.node:
        layers.append(make_layer(name=node.name or node.op_type, type_name=node.op_type))

    for out in graph.output:
        layers.append(make_layer(name=out.name, type_name="Output", extra=_shape_str(out)))

    return layers
