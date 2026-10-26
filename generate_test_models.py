"""Generate one test model per supported format so the parsers can be
validated end-to-end without needing external downloaded model files."""
import json
import os
import struct

import h5py
import torch
import torch.nn as nn

OUT = os.path.join(os.path.dirname(__file__), "test_fixtures")
os.makedirs(OUT, exist_ok=True)


# ---------- 1. PyTorch: a small CRNN (CNN + GRU) full module ----------
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.gru = nn.GRU(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        b, c, h, w = x.shape
        x = x.view(b, h * w, c)
        x, _ = self.gru(x)
        return self.fc(x[:, -1])


model = CRNN()
torch.save(model, os.path.join(OUT, "crnn_full.pt"))
torch.save(model.state_dict(), os.path.join(OUT, "crnn_state_dict.pt"))
print("wrote crnn_full.pt, crnn_state_dict.pt")

# scripted TorchScript archive too
scripted = torch.jit.script(model)
scripted.save(os.path.join(OUT, "crnn_scripted.pt"))
print("wrote crnn_scripted.pt")


# ---------- 2. ONNX: export a small CNN ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        return self.fc(x.flatten(1))


cnn = SimpleCNN()
dummy = torch.randn(1, 3, 32, 32)
torch.onnx.export(cnn, dummy, os.path.join(OUT, "simple_cnn.onnx"),
                   input_names=["input"], output_names=["output"], opset_version=13)
print("wrote simple_cnn.onnx")


# ---------- 3. Fake Keras .h5 (model_config only, no real TF needed) ----------
keras_config = {
    "class_name": "Sequential",
    "config": {
        "name": "sequential",
        "layers": [
            {"class_name": "Embedding", "config": {"name": "embedding"}},
            {"class_name": "LSTM", "config": {"name": "lstm_1"}},
            {"class_name": "Dropout", "config": {"name": "dropout_1"}},
            {"class_name": "Dense", "config": {"name": "dense_out"}},
        ],
    },
}
with h5py.File(os.path.join(OUT, "fake_keras_lstm.h5"), "w") as f:
    f.attrs["model_config"] = json.dumps(keras_config)
    f.create_dataset("dummy", data=[1, 2, 3])
print("wrote fake_keras_lstm.h5")


# ---------- 4. Synthetic .safetensors (header only matters for our parser) ----------
header = {
    "encoder.conv1.weight": {"dtype": "F32", "shape": [16, 3, 3, 3], "data_offsets": [0, 0]},
    "encoder.conv1.bias": {"dtype": "F32", "shape": [16], "data_offsets": [0, 0]},
    "transformer.layer0.attn.weight": {"dtype": "F32", "shape": [64, 64], "data_offsets": [0, 0]},
    "classifier.fc.weight": {"dtype": "F32", "shape": [10, 64], "data_offsets": [0, 0]},
    "__metadata__": {"format": "pt"},
}
header_bytes = json.dumps(header).encode("utf-8")
with open(os.path.join(OUT, "fake_transformer.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(header_bytes)))
    f.write(header_bytes)
print("wrote fake_transformer.safetensors")

print("\nAll test fixtures created in", OUT)
