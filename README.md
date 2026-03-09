In this, i focused on optimizing machine learning inference for production environments rather than improving model accuracy alone. A baseline LSTM model is trained on historical stock data and then optimized for CPU deployment using quantization and ONNX Runtime. The objective is to reduce model size, lower inference latency, and increase throughput while keeping the model architecture unchanged.

Stack:
-Modeling: PyTorch (LSTM)
-Inference Optimization: ONNX Runtime, PyTorch Dynamic Quantization
-Data Pipeline: Yahoo Finance (yfinance), Scikit-Learn
-Monitoring: Concept drift detection using rolling statistical checks

Tools:
PyTorch
ONNX Runtime
NumPy
Scikit-Learn
yfinance

to see it working
-Install dependencies: 
pip install -r requirements.txt
-Train the baseline model: 
python train_baseline.py
-Optimize the model for inference: 
python optimize_model.py
-Run performance benchmarks: 
python benchmark.py
-Run drift monitoring: 
python drift_monitor.py


