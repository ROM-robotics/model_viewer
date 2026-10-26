"""Parse Keras/TensorFlow model files into a LayerInfo sequence.

Supports:
  - Legacy HDF5 ``.h5`` files saved with ``model.save(...)`` (reads the
    ``model_config`` JSON attribute directly via h5py, no TensorFlow needed).
  - Native Keras 3 ``.keras`` files, which are zip archives containing a
    ``config.json`` with the same architecture structure.

Weights-only HDF5 files (no architecture metadata) fall back to guessing
layer types from the dataset/group names, same as a raw state_dict.
"""
import json
import zipfile

import h5py

from .common import make_layer, layers_from_tensor_names


def _walk(node, layers):
    class_name = node.get("class_name")
    config = node.get("config", {}) or {}
    if class_name in ("Sequential", "Functional", "Model") and "layers" in config:
        for sub in config["layers"]:
            _walk(sub, layers)
    else:
        layers.append(make_layer(name=config.get("name", class_name), type_name=class_name or "Unknown"))


def _parse_config_json(config_text: str):
    cfg = json.loads(config_text)
    layers = []
    _walk(cfg, layers)
    return layers


def _parse_h5(path: str):
    with h5py.File(path, "r") as f:
        config_text = f.attrs.get("model_config")
        if config_text is None:
            # Weights-only file: guess layers from the dataset names instead.
            names = []

            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    names.append(name)

            f.visititems(collect)
            if not names:
                raise ValueError("H5 file has no model_config and no weight datasets")
            return layers_from_tensor_names(names)

        if isinstance(config_text, bytes):
            config_text = config_text.decode("utf-8")
        return _parse_config_json(config_text)


def _parse_keras_zip(path: str):
    with zipfile.ZipFile(path) as zf:
        with zf.open("config.json") as f:
            config_text = f.read().decode("utf-8")
    return _parse_config_json(config_text)


def parse(path: str):
    if zipfile.is_zipfile(path):
        return _parse_keras_zip(path)
    return _parse_h5(path)
