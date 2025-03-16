import pandas as pd
import glob
import numpy as np
from pandas import DataFrame


def clean_df(df_input: pd.DataFrame) -> DataFrame | None:
    df_input["forceZ"] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    if len(df_input) == 0:
        return None

    new_df = pd.DataFrame({
        "posx": df_input['posx'] + df_input['posx_d'] / 1000,
        "posy": df_input['posy'] + df_input['posy_d'] / 1000,
        "posz": df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000,
        "Fz": df_input['forceZ'],
        "t": df_input['CPXEts'],
        "isTouching_SMAC": df_input['isTouching_SMAC'],
        "isArrived_Festo": df_input['isArrived_Festo'],
    })

    return new_df


# Return only the signal captured when the machine is touching
def get_df_list(path: str = 'Dataset/20250205_082609_HIST_006_CPXE_*.csv') -> list[pd.DataFrame]:
    # Get list of all CSV files
    csv_files = glob.glob(path)

    df_list = list()
    for file in csv_files:
        data_frame = pd.read_csv(file)
        df_clean = clean_df(data_frame)

        mean = np.array(df_clean[df_clean['isArrived_Festo'] == 1]["Fz"])[:50].mean()  # initial offset
        df_clean["Fz"] = df_clean["Fz"] - mean

        df_clean = df_clean[df_clean['isTouching_SMAC'] == 1].copy()
        df_clean["t"] = df_clean["t"] - df_clean['t'].min()

        if df_clean is not None and len(df_clean) > 0:
            df_list.append(df_clean)

    return df_list


def get_df_list2(path: str = 'Dataset/20250205_082609_HIST_006_CPXE_*.csv') -> list[pd.DataFrame]:
    # Get list of all CSV files
    csv_files = glob.glob(path)

    df_list = list()
    for file in csv_files:
        data_frame = pd.read_csv(file)
        df_clean = clean_df(data_frame)

        mean = np.array(df_clean[df_clean['isArrived_Festo'] == 1]["Fz"])[:50].mean()  # initial offset
        df_clean["Fz"] = df_clean["Fz"] - mean

        df_clean = df_clean[df_clean['isArrived_Festo'] == 1].copy()
        df_clean["t"] = df_clean["t"] - df_clean['t'].min()

        df_clean2 = df_clean[df_clean['isTouching_SMAC'] == 1].copy()

        if df_clean2 is not None and len(df_clean2) > 0:
            df_list.append(df_clean)

    return df_list
