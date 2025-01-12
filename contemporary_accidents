import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape X for LSTM input (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Convert to float32
X_train_reshaped = X_train_reshaped.astype(np.float32)
X_val_reshaped = X_val_reshaped.astype(np.float32)
X_test_reshaped = X_test_reshaped.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

# Define the LSTM model using Input layer for the input shape
model = Sequential()
model.add(Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))  # Define input shape here
model.add(LSTM(25, activation='tanh'))  # LSTM layer now doesn't need input_shape
# model.add(Dropout(0.2))  # Uncomment if dropout is needed
model.add(Dense(1))

# # Compile the model
model.compile(optimizer='sgd', loss='mean_absolute_error')

# from tensorflow.keras.layers import Bidirectional

# model = Sequential()
# model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
# model.add(Dropout(0.5))
# model.add(Bidirectional(LSTM(32, activation='tanh')))
# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mean_absolute_error')


# Train the model using validation data
history = model.fit(X_train_reshaped, y_train, epochs=150, batch_size=32, 
                    validation_data=(X_val_reshaped, y_val), verbose=1)

# Make predictions
y_train_pred = model.predict(X_train_reshaped).flatten()
y_val_pred = model.predict(X_val_reshaped).flatten()
y_test_pred = model.predict(X_test_reshaped).flatten()

# y_train_pred = np.where(y_train_pred < 0, 0, np.where(y_train_pred > 100, 100, y_train_pred))
# y_val_pred = np.where(y_val_pred < 0, 0, np.where(y_val_pred > 100, 100, y_val_pred))
# y_test_pred = np.where(y_test_pred < 0, 0, np.where(y_test_pred > 100, 100, y_test_pred))

# Evaluate the model
train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)


# Print evaluation metrics
print("\nModel Performance Metrics")
print("-------------------------")
print(f"MAE : Training: {train_mae:.4f}, Validation: {val_mae:.4f}, Test: {test_mae:.4f}")
