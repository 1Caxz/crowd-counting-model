import torch
import torch.nn as nn
from models.csrnet_mbv3 import CSRNetMobile
import os

# --- Load model ---
model = CSRNetMobile()
model.load_state_dict(torch.load("csrnet_mobile_kd.pt", map_location=torch.device('cpu')))
model.eval()

# --- Trace dummy input ---
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)
torchscript_path = "csrnet_mobile_scripted.pt"
traced_model.save(torchscript_path)
print(f"✅ TorchScript saved to {torchscript_path}")

# --- ONNX export (optional step before TFLite conversion) ---
onnx_path = "csrnet_mobile.onnx"
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                  opset_version=11)
print(f"✅ ONNX model exported to {onnx_path}")

# Note: TFLite conversion must be done using ONNX → TF → TFLite via Python tools like tf2onnx + tf.lite
# Example (to run separately):
#   python -m tf2onnx.convert --onnx csrnet_mobile.onnx --saved-model tf_model
#   then use TF to convert tf_model to .tflite

# Tip: In Flutter, use "tflite_flutter" to run .tflite model