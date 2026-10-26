# AI Model Viewer

Upload an AI model file (`.pt` / `.pth` / `.onnx` / `.h5` / `.keras` / `.safetensors`)
and see its architecture two ways:

- **Simplified (Mermaid)** — layers are grouped into color-coded blocks
  (Convolution, Normalization, Recurrent/LSTM/GRU, Attention, Linear, ...)
  flowing from Input → ... → Output, with an auto-detected architecture
  label (CNN / Gated RNN / Transformer / hybrid combinations).
- **Detailed (Netron)** — the full computational graph, powered by the
  [Netron](https://github.com/lutzroeder/netron) open-source viewer,
  embedded in an iframe.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# torch's CPU wheel isn't on PyPI's default index:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
source venv/bin/activate
python app.py
```

Then open **http://127.0.0.1:5000** in a browser and upload a model file.

## Try it with sample models

`generate_test_models.py` creates one example file per supported format
(a small CNN+GRU hybrid, a plain CNN exported to ONNX, a fake Keras LSTM
config, and a synthetic Transformer-ish safetensors file) under
`test_fixtures/`, useful for quickly trying out the viewer:

```bash
source venv/bin/activate
python generate_test_models.py
```

## How layer types are detected per format

| Format | Source of layer info |
|---|---|
| `.pt` / `.pth` (TorchScript) | `named_modules()` of the loaded scripted module |
| `.pt` / `.pth` (full pickled `nn.Module`) | `named_modules()` (requires the original model class to be importable) |
| `.pt` / `.pth` (state_dict only) | Layer names guessed from parameter-name prefixes (e.g. `conv1.weight` → Conv) |
| `.onnx` | The ONNX graph's node `op_type`s directly |
| `.h5` / `.keras` | The `model_config` JSON Keras stores alongside weights |
| `.safetensors` | Layer names guessed from the tensor-name header, same heuristic as a raw state_dict |

**Known limitation:** state_dict-only `.pt`/`.pth` files and `.safetensors`
files carry weights but no architecture graph, so layer *types* for those
are a best-effort guess based on naming conventions, not guaranteed exact.
ONNX models that reference an external weights file (e.g. `model.onnx.data`)
parse fine for the Simplified view (only the graph structure is read), but
the Detailed Netron view may be incomplete if that sibling file wasn't
uploaded alongside the model.

## Project structure

```
app.py                 Flask routes (upload, analyze, render)
netron_wrapper.py       Launches a local Netron server per uploaded model
parsers/
  common.py             Shared LayerInfo model + category heuristics
  pytorch_parser.py      .pt / .pth
  onnx_parser.py         .onnx
  keras_parser.py        .h5 / .keras
  safetensors_parser.py  .safetensors
  architecture.py        CNN/RNN/Gated/Transformer classification
  mermaid_gen.py         Mermaid flowchart generator
templates/              index.html (upload form), result.html (diagrams)
```
# model_viewer
