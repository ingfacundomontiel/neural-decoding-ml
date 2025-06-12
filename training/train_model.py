import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from models.MLP import MLP

# === CONFIGURATION ===
INPUT_SIZE = 100           # To be replaced with actual feature size
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1            # Single continuous value (e.g., position)
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# === LOAD DATA ===
# Replace with actual data loading logic
X = np.load('datasets/processed-datasets/X.npy')
y = np.load('datasets/processed-datasets/y.npy')

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Add dimension for output

dataset = TensorDataset(X_tensor, y_tensor)
val_size = int(len(dataset) * VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === INITIALIZE MODEL ===
model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === TRAINING LOOP ===
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# === EVALUATION ===
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X)
        y_true.append(batch_y.numpy())
        y_pred.append(outputs.numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"\nFinal Evaluation:\nRMSE = {rmse:.4f}\nR^2 Score = {r2:.4f}")
