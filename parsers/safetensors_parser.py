"""Parse .safetensors files into a LayerInfo sequence.

The format is: 8-byte little-endian header length, then a JSON header
mapping tensor name -> {dtype, shape, data_offsets}, followed by raw tensor
bytes. We only need the header, so no safetensors/torch dependency is
required to read it, and it's cheap even for multi-GB weight files.
"""
import json
import struct

from .common import layers_from_tensor_names


def parse(path: str):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    names = [k for k in header.keys() if k != "__metadata__"]
    if not names:
        raise ValueError("safetensors file has no tensors")

    shape_lookup = {}
    for name in names:
        shape = header[name].get("shape")
        if shape:
            shape_lookup[name] = "x".join(str(d) for d in shape)

    return layers_from_tensor_names(names, shape_lookup)
