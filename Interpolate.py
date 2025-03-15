import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def cubic_interpolation_padded(list_dataframes: list[pd.DataFrame], period: float) -> list[pd.DataFrame]:
    interpolated_list = []

    for element in list_dataframes:
        # Create cubic interpolation function
        interp_func1 = interp1d(element['t'], element['posz'], kind='cubic')
        interp_func2 = interp1d(element['t'], element['Fz'], kind='cubic')

        # Define new time points with a constant interval
        t_new = np.arange(element['t'].min(), element['t'].max(), period)  # Adjust step size as needed
        posz_new = interp_func1(t_new)
        fz_new = interp_func2(t_new)

        # Create new DataFrame with interpolated values
        df_interp = pd.DataFrame({'t': t_new, 'posz': posz_new, 'Fz': fz_new})
        interpolated_list.append(df_interp)
    return pad_time_series(interpolated_list)


def pad_time_series(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    # max_length = max(len(df) for df in dataframes)  # Find longest series
    lengths = sorted(set(len(df) for df in dataframes), reverse=True)  # Get unique lengths in descending order
    second_max_length = lengths[1] if len(lengths) > 1 else None  # Get second maximum if available

    padded_dfs = []
    for df in dataframes:
        pad_size = second_max_length - len(df)

        if pad_size > 0:  # If this series is shorter, pad it
            last_row = df.iloc[-1]  # Get last row values
            pad_df = pd.DataFrame([last_row] * pad_size)  # Repeat last row
            df = pd.concat([df, pad_df], ignore_index=True)  # Append padding

            padded_dfs.append(df)

    return padded_dfs
