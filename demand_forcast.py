import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')

# Feature Engineering: Add moving average, lag features, and momentum indicators
df['MA7'] = df['Demand'].rolling(window=7).mean()
df['Lag1'] = df['Demand'].shift(1)
df['Momentum'] = df['Demand'] - df['Demand'].shift(3)
df.dropna(inplace=True)

# Data Normalization
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Prepare Sequences for LSTM
X, y = [], []
seq_length = 10
for i in range(len(df_scaled) - seq_length):
    X.append(df_scaled[i:i + seq_length])
    y.append(df_scaled[i + seq_length, 0])
X, y = np.array(X), np.array(y)

# Convert target to probability-like score (normalized demand)
y = (y - y.min()) / (y.max() - y.min())  # Scale to 0-1 for sigmoid activation

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Build Improved LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Predicts probability (0 to 1)
])

# Compile Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')

# Train Model with Early Stopping & Learning Rate Scheduler
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, lr_scheduler])

# Predict Next Day
last_seq = X[-1].reshape(1, seq_length, X.shape[2])
predicted_prob = model.predict(last_seq)[0][0] * 100  # Convert to percentage

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(y_test):], y_test * 100, label='Actual Demand Probability (%)', color='blue')
plt.scatter(df.index[-1] + pd.Timedelta(days=1), predicted_prob, color='red', label='Predicted Demand (%)', s=100)
plt.legend()
plt.title("Actual vs Predicted Demand Probability")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

print(f"Predicted Demand Probability for next day: {predicted_prob:.2f}%")
