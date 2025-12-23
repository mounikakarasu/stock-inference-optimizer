Stock Inference Engine
This project is not just a stock predictor, it is an inference optimizer Engine. While many projects solely focus on training accuracy, my project focuses on production-grade engineering.
I built a standard LSTM pipeline and then optimized it for edge deployment using Quantization and ONNX Runtime, achieving a 4x reduction in model size and significant latency improvements on CPU hardware.

Key Engineering Achievements

* Model size reduced by ~75%. Optimized from ~350 KB (PyTorch FP32) to ~90 KB using ONNX INT8 quantization.
* Inference latency improved ~4x. Reduced from ~5.0 ms to ~1.2 ms per request on CPU
* Throughput increased ~4x. Scaled from ~200 requests/sec to ~800 requests/sec under indentical conditions.

note: benchmarks were run on a standard consumer grade CPU and gains were achieved through model export and quantization, not architectural changes.

Tech stack and Optimization Techniques
Modelling - PyTorch (LSTM)
Inference Acceleration - ONNX Runtime, PyTorch Dynamic Quantization
Ops \& Monitoring - Concept drift detection (rolling window statistical checks)
Data Pipeline - Yahoo Finance (yfinance), Scikit-Learn

Optimization Pipeline

* baseline training: trained a 2 layer LSTM on AAPL historical data
* dynamic quantization: converted FP32 weights to INT8, reducing memory footprint by 4x with negligible accuracy loss.
* graph compilation: exported to Open Neural Network Exchange to fuse layers and eliminate Python overhead (GIL)
* drift monitoring: implemented a real-time monitor that triggers alerts when prediction error variance exceeds a safety threshold.





Folder Structure  
d-----venv
-a----benchmark.py
-a----drift\_monitor.py
-a----optimize\_model.py
-a----README.md
-a----train\_baseline.py

