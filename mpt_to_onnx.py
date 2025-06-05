import torch
import torch.onnx
from models.csrnet_mbv3 import CSRNetMobile

# Load model
model = CSRNetMobile()
model.load_state_dict(torch.load("csrnet_mobile.pt"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "csrnet_mobile.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=11)

print("✅ Converted to ONNX: csrnet_mobile.onnx")
