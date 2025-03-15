from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def create_input_signals(df_list: list[pd.DataFrame]) -> np.ndarray:
    input_signals = []

    for df in df_list:
        signal = np.column_stack((df['posz'], df['Fz']))

        if signal.shape != (702, 2):
            print(signal.shape)

            # Append this signal to the list of input signals
        input_signals.append(signal)

    # Convert list to numpy array with shape (n_samples, 502, 2)
    return np.array(input_signals)


def preprocess_signals(df_list, scaler: StandardScaler):
    # Extract all signals first
    all_signals = create_input_signals(df_list)

    # Reshape to 2D for scaling (samples*timestep, features)
    original_shape = all_signals.shape
    reshaped = all_signals.reshape(-1, original_shape[-1])

    # Fit scaler and transform
    scaled = scaler.fit_transform(reshaped)

    # Reshape back to original shape
    normalized_signals = scaled.reshape(original_shape)

    return normalized_signals, all_signals  # Return scaler to inverse transform later
