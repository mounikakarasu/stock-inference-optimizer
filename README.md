In this, i focused on optimizing machine learning inference for production environments rather than improving model accuracy alone. A baseline LSTM model is trained on historical stock data and then optimized for CPU deployment using quantization and ONNX Runtime. The objective is to reduce model size, lower inference latency, and increase throughput while keeping the model architecture unchanged.

Stack:
Modeling
PyTorch (LSTM)
Inference Optimization
ONNX Runtime
PyTorch Dynamic Quantization
Data Pipeline
Yahoo Finance (yfinance)
Scikit-Learn
Monitoring
Concept drift detection using rolling statistical checks

Tools:
PyTorch
ONNX Runtime
NumPy
Scikit-Learn
yfinance

to see it working
-Install dependencies
-Train the baseline model
-Optimize the model for inference
-Run performance benchmarks
-Run drift monitoring



