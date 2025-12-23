import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import os

#config
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'
SEQ_LENGTH = 60    # Look back 60 days
HIDDEN_SIZE = 128  # Intentionally "bloated" for optimization demo
LAYERS = 2
EPOCHS = 5

#pipeline
print(f"Downloading data for {TICKER}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

if len(data) == 0:
    raise ValueError("No data fetched! Check your internet connection.")

#using close price
prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled_prices, SEQ_LENGTH)

# Convert to PyTorch Tensors
X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).float()

print(f"Data prepared. Shape: {X_train.shape}")

#heavymodel
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYERS, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take the last time step
        return out

#trainingloop
model = StockLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.5f}")

duration = time.time() - start_time
print(f"Training finished in {duration:.2f} seconds.")

torch.save(model.state_dict(), "baseline_model.pth")
print(f"Model saved to: {os.path.abspath('baseline_model.pth')}")