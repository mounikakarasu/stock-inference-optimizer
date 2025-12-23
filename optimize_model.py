import torch
import torch.quantization
import torch.onnx
import torch.nn as nn
import os


#structure-redifining class to load weights
HIDDEN_SIZE = 128
LAYERS = 2

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYERS, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#baselineloading
model_path = "baseline_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Run train_baseline.py first! Could not find {model_path}")

print(f"Loading {model_path}...")
model = StockLSTM()
model.load_state_dict(torch.load(model_path))
model.eval() # Set to evaluation mode (crucial for inference)

#optimizing using dynamic quantizationINT8
print("Applying Dynamic Quantization (Float32 -> INT8)...")
# We only quantize the heavy layers (LSTM and Linear)
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.LSTM, torch.nn.Linear}, 
    dtype=torch.qint8
)

save_path_quant = "quantized_model.pth"
torch.save(quantized_model.state_dict(), save_path_quant)
print(f"Saved: {save_path_quant}")

#optimizing using ONNX Export
print("Exporting to ONNX format...")
save_path_onnx = "model.onnx"

#dummyinput to trace paragraph
dummy_input = torch.randn(1, 60, 1)

torch.onnx.export(
    model, 
    dummy_input, 
    save_path_onnx,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)
print(f"Saved: {save_path_onnx}")

#verifying
def get_file_size(path):
    return os.path.getsize(path) / 1024 # Convert bytes to KB

size_base = get_file_size(model_path)
size_quant = get_file_size(save_path_quant)
size_onnx = get_file_size(save_path_onnx)

print("\n results")
print(f"Original Model:   {size_base:.2f} KB")
print(f"Quantized Model:  {size_quant:.2f} KB  (Should be ~4x smaller)")
print(f"ONNX Model:       {size_onnx:.2f} KB")
print("-" * 30)