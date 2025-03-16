import pandas as pd
import glob
import numpy as np
import json
from pandas import DataFrame
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


def exponential_decay(t, A, tau, B):
    return A * np.exp(-t / tau) + B


def extract_features(data):
    # Gaussian Smoothing
    data['Fz_s'] = gaussian_filter1d(data['forceZ'], sigma=2)
    data['posz_s'] = gaussian_filter1d(data['posz_corrected'], sigma=2)

    pz = (data['posz_s'][data['isArrived_Festo'] == 1] / 1000).to_numpy()  # From mm to m
    fz = (data['Fz_s'][data['isArrived_Festo'] == 1]).to_numpy()
    time = data['CPXEts'][data['isArrived_Festo'] == 1].to_numpy()

    fz_touch = (data['Fz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    time_touch = data['CPXEts'][data['isTouching_SMAC'] == 1].to_numpy()

    # Force Decay Rate (τ)
    popt, _ = curve_fit(exponential_decay, time_touch, fz_touch, p0=[fz_touch[0] - fz_touch[-1], 0.4, fz_touch[-1]])
    tau = popt[1]  # Decay rate

    # Steady-State Force (F_ss) - Mean force at end of signal
    F_ss = np.mean(fz_touch[-20:])  # Last 10 samples

    # Stiffness
    stiffness = np.max(data['Fz_s']) / np.max(data['posz_s'])

    # AUCs (areas under the curves)
    area_f = simpson(fz, x=time)
    area_p = simpson(pz, x=time)

    # Dominant Frequency (from FFT)
    fft_vals = np.abs(np.fft.rfft(fz))
    fft_f = np.fft.rfftfreq(len(time), d=np.mean(np.diff(time)))
    dom_f = fft_f[np.argmax(fft_vals)]

    # Slope of Initial Step Up & Final Step Down
    gradient = np.gradient(fz, time)
    upstroke = np.max(gradient)  # Max increase
    downstroke = np.min(gradient)  # Max decrease

    # Power
    power = np.mean(fz ** 2)

    # Entropy of the Force Signal
    prob_dist = np.histogram(data['forceZ'][data['isArrived_Festo'] == 1], bins=50, density=True)[
        0]  # Probability distribution
    prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
    entropy = -np.sum(prob_dist * np.log2(
        prob_dist))  # Shannon entropy => measure of randomness (estimation of fluctuations in the signal)

    return stiffness, tau, F_ss, power, entropy, area_f, area_p, dom_f, upstroke, downstroke


def find_inclusions(json_data):
    circles = json_data["Inclusions"]
    c = [(circle["Position"][0] + 50, circle["Position"][1] + 50) for circle in circles]
    r = [circle["Diameter"] / 2 for circle in circles]
    return c, r


def get_label(posx, posy, centers, radii) -> int:
    for i in range(len(centers)):
        center_x, center_y = centers[i]
        radius = radii[i]
        # Check if point is within the circle
        if ((posx - center_x) ** 2 + (posy - center_y) ** 2) <= radius ** 2:
            return (i // 5) + 1  # Assign label based on group of 5 circles
    return 0  # Default label


def organize_df(df_input: DataFrame, centers, radii) -> DataFrame | None:
    df_input['forceZ'] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    offset = np.mean(df_input['forceZ'][df_input['isArrived_Festo'] == 1].head(20))
    df_input['forceZ'] = (df_input['forceZ'] - offset)
    df_input['forceZ'] = df_input['forceZ'] / df_input['forceZ'].max()  # normalized and corrected
    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    posx = np.mean(df_input['posx'][df_input['isArrived_Festo'] == 1] + df_input['posx_d'][
        df_input['isArrived_Festo'] == 1] / 1000)
    posy = np.mean(df_input['posy'][df_input['isArrived_Festo'] == 1] + df_input['posy_d'][
        df_input['isArrived_Festo'] == 1] / 1000)

    df_input['posz_corrected'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000

    stiffness, tau, F_ss, power, entropy, area_f, area_p, dom_f, upstroke, downstroke = extract_features(df_input)

    label = get_label(posx, posy, centers, radii)

    new_df = DataFrame({
        "posx": posx,
        "posy": posy,
        "posz": [df_input['posz_corrected'].tolist()],
        "Stiffness": stiffness,
        "Tau": tau,
        "Force Steady State": F_ss,
        "Power": power,
        "Entropy": entropy,
        "F_area": area_f,
        "P_area": area_p,
        "Dominant Frequency": dom_f,
        "Upstroke": upstroke,
        "Downstroke": downstroke,
        "Fz": [df_input['forceZ'].tolist()],
        "t": [df_input['CPXEts'].tolist()],
        "Fz_s": [df_input['Fz_s'].tolist()],
        "posz_s": [df_input['posz_s'].tolist()],
        "Touching": [df_input['isTouching_SMAC'].tolist()],
        "label": label
    })

    return new_df


def create_df(path: str = 'Dataset/20250205_082609_HIST_006_CPXE_*.csv') -> DataFrame | None:
    # Load JSON file
    with open("phantom_metadata.json", "r") as file:
        json_data = json.load(file)

    # Create lists of centers + radii (for each labeled area)
    centers, radii = find_inclusions(json_data)

    # Get list of all CSV files
    csv_files = glob.glob(path)

    df_list = list()

    for file in csv_files:
        df = pd.read_csv(file)
        no_touch = (df['isTouching_SMAC'] == 0).all()
        if no_touch: continue
        df_ta = organize_df(df, centers, radii)
        df_ta.insert(0, "Source", file.split("_")[-1].split(".")[0])
        if df_ta is not None and len(df_ta) > 0:
            df_list.append(df_ta)

    return pd.concat(df_list, ignore_index=True)
