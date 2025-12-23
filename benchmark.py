import torch
import torch.quantization
import onnxruntime as ort
import time
import numpy as np
import os
from train_baseline import StockLSTM 

#config
input_shape = (1, 60, 1) 
n_warmup = 10             
n_loops = 500             

print(f"Benchmarking setup: Warmup={n_warmup}, Loops={n_loops}")
print("-" * 60)

#input
dummy_input_torch = torch.randn(input_shape)
dummy_input_numpy = dummy_input_torch.numpy()

#loading models

#baseline
model_base = StockLSTM()
# FIX: Added weights_only=False to support older/custom formats safely
model_base.load_state_dict(torch.load("baseline_model.pth", weights_only=False))
model_base.eval()

#quantized model
model_quant_struct = StockLSTM()
model_quant = torch.quantization.quantize_dynamic(
    model_quant_struct, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
)

model_quant.load_state_dict(torch.load("quantized_model.pth", weights_only=False))
model_quant.eval()

#ONNX runtime
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess_options.log_severity_level = 3 
ort_session = ort.InferenceSession("model.onnx", sess_options)

#benchmark
def measure_latency(name, inference_func, input_data):
    for _ in range(n_warmup): inference_func(input_data)
    
    start_time = time.time()
    for _ in range(n_loops): inference_func(input_data)
    end_time = time.time()
    
    avg_ms = ((end_time - start_time) / n_loops) * 1000
    return avg_ms

def run_torch(x):
    with torch.no_grad(): return model_base(x)

def run_quant(x):
    with torch.no_grad(): return model_quant(x)

def run_onnx(x):
    return ort_session.run(None, {'input': x})

print("\nRunning Benchmarks")
latency_base = measure_latency("PyTorch FP32", run_torch, dummy_input_torch)
latency_quant = measure_latency("PyTorch INT8", run_quant, dummy_input_torch)
latency_onnx = measure_latency("ONNX Runtime", run_onnx, dummy_input_numpy)

print(f"\n{'Model Version':<20} | {'Latency (ms)':<15} | {'Speedup'}")
print("-" * 55)
print(f"{'Baseline (FP32)':<20} | {latency_base:.4f} ms      | 1.0x (Ref)")
print(f"{'Quantized (INT8)':<20} | {latency_quant:.4f} ms      | {latency_base/latency_quant:.2f}x")
print(f"{'ONNX Runtime':<20} | {latency_onnx:.4f} ms      | {latency_base/latency_onnx:.2f}x")
print("-" * 55)