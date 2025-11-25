import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Data Preparation
df = pd.read_csv("train_cleansed.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True, errors='coerce')
monthly_data = df.resample('ME', on='Order Date').sum()

all_data = monthly_data['Sales'].values
train_data = all_data[:-12] 
test_data = all_data[-24:-12]
y_test = all_data[-12:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler.transform(test_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)

def create_inout_sequences(input_data, window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - window):
        train_seq = input_data[i:i+window]
        train_label = input_data[i+window:i+window+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

inout_seq = create_inout_sequences(train_data_normalized, 12)

# LSTM Model Definition
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=48, num_layers=3, output_size=1, dropout=0.0):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out[-1])
        return predictions

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# Training
for i in range(200):
    for seq, labels in inout_seq:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Predicting
test_input = test_data_normalized[-12:].tolist()
model.eval()
for i in range(12):
    seq = torch.FloatTensor(test_input[-12:])
    with torch.no_grad():
        yhat = model(seq)
        test_input.append(yhat.item())

prediction2 = scaler.inverse_transform(np.array(test_input[12:]).reshape(-1, 1))

monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()
X_test = pd.date_range(start=monthly_sales.index[-12], periods=12, freq='ME')

y_pred2 = pd.Series(prediction2[:, 0], index=X_test)

# Plot
ax = monthly_sales.plot(style="-.", color="0.5", title="Predicting Using LSTM")
y_pred2.plot(ax=ax, linewidth=3, label="LSTM Forecast", color='C2')
plt.legend()
plt.show()

# Metrics and evaluation
print("R² test:", r2_score(y_test, prediction2[:, 0]))
# Add here your evaluate_model function if you want more metrics shown:
# evaluate_model('LSTM', y_test, y_pred2)
