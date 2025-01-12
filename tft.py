import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape X for TFT input (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Convert to float32
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

# Transformer Block
def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    # Multi-Head Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = Add()([inputs, attn_output])  # Residual Connection
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

    # Feed Forward Network
    ff_output = Dense(ff_dim, activation='relu')(attn_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Add()([attn_output, ff_output])  # Residual Connection
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

    return ff_output

# Build the TFT Model
input_layer = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
transformer_output = transformer_block(input_layer, num_heads=4, ff_dim=64)
transformer_output = transformer_block(transformer_output, num_heads=4, ff_dim=64)
transformer_output = GlobalAveragePooling1D()(transformer_output)
output_layer = Dense(1)(transformer_output)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
history = model.fit(
    X_train_reshaped, y_train,
    epochs=150, batch_size=32,
    validation_data=(X_val_reshaped, y_val),
    verbose=1
)

# Make predictions
y_train_pred = model.predict(X_train_reshaped).flatten()
y_val_pred = model.predict(X_val_reshaped).flatten()
y_test_pred = model.predict(X_test_reshaped).flatten()

# Evaluate the model
train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)


# Print evaluation metrics
print("\nModel Performance Metrics")
print("-------------------------")
print(f"MAE : Training: {train_mae:.4f}, Validation: {val_mae:.4f}, Test: {test_mae:.4f}")

