import pandas as pd
import glob
import numpy as np
import json
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.signal import welch
from scipy.interpolate import interp1d


def extract_features(data):
    # Gaussian Smoothing
    data['Fz_s'] = gaussian_filter1d(data['forceZ'], sigma=2)
    data['posz_s'] = gaussian_filter1d(data['posz'], sigma=2)

    pz = (data['posz_s'][data['isArrived_Festo'] == 1] / 1000).to_numpy()  # From mm to m
    fz = (data['Fz_s'][data['isArrived_Festo'] == 1]).to_numpy()
    time = data['CPXEts'][data['isArrived_Festo'] == 1].to_numpy()

    fz_touch = (data['Fz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    pz_touch = (data['posz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    time_touch = data['CPXEts'][data['isTouching_SMAC'] == 1].to_numpy()

    time_i = np.arange(time[0], time[-1], 0.001)
    f = interp1d(time, fz, kind='cubic')

    fi = f(time_i)
    g = interp1d(time, pz, kind='cubic')
    pi = g(time_i)
    # Force Decay Rate (τ)
    decay_time = time_touch - time_touch[0]
    decay_force = np.log(fz_touch)  # Log transform to get a linear function like ln(F) = -t/tau + c
    tau, _, _, _, _ = linregress(decay_time, decay_force)
    tau = -1 / tau  # Convert to time constant

    # Steady-State Force (F_ss) - Mean force at end of signal
    F_ss = np.mean(fz_touch[-30:])
    P_ss = np.mean(pz_touch[-30:])

    # Hardness
    start_idx = np.where(fz >= 0.1)[0][0]
    stiffness = (fz.max() - 0.1) / (pz[fz.argmax()] - pz[start_idx])

    idx1 = np.where(data['Fz_s'] > 0.5)[0][0]  # First index where force > 0.5
    idx2 = np.where(data['Fz_s'] > 0.1)[0][0]  # First index where force > 0.1
    upstroke = (data['Fz_s'][idx1] - data['Fz_s'][idx2]) / (data['CPXEts'][idx1] - data['CPXEts'][idx2])

    idx1 = np.where(data['Fz_s'] > 0.5)[0][-1]  # First index where force > 0.5
    idx2 = np.where(data['Fz_s'] > 0.1)[0][-1]  # First index where force > 0.1
    downstroke = (data['Fz_s'][idx1] - data['Fz_s'][idx2]) / (data['CPXEts'][idx1] - data['CPXEts'][idx2])

    offset = np.mean(fz_touch[:5]) - np.mean(fz_touch[:-10])

    # PSD
    freqs, psd = welch(fz, fs=1 / np.median(np.diff(time)))
    power = np.mean(np.square(fz_touch)) / (time_touch[-1] - time_touch[0])

    psd_peak = freqs[np.argmax(psd)]

    # Entropy of the Force Signal
    prob_dist = np.histogram(data['Fz_s'][data['isTouching_SMAC'] == 1], bins=30, density=True)[
        0]  # Probability distribution
    prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
    entropy = -np.sum(prob_dist * np.log2(
        prob_dist))  # Shannon entropy => measure of randomness (estimation of fluctuations in the signal)

    # NEW FEATURES

    # 1. Initial Force Peak Characteristics
    force_max = np.max(fz)
    force_max_idx = np.argmax(fz)
    time_to_max = time[force_max_idx] - time[0]

    # 2. Force Overshoot - This seems prominent in your graphs
    steady_state_start_idx = force_max_idx + int(len(fz) * 0.1)  # Skip a bit after peak
    steady_state_force = np.mean(fz[steady_state_start_idx:steady_state_start_idx + 100])
    force_overshoot = (force_max - steady_state_force) / steady_state_force

    # 3. Initial Peak Width - Different materials show different peak widths
    half_height = (force_max - fz[0]) / 2 + fz[0]
    above_half_height = np.where(fz > half_height)[0]
    peak_width = time[above_half_height[-1]] - time[above_half_height[0]]

    # 4. Force-Position Hysteresis Area
    # Create force vs position curves for loading and unloading phases
    midpoint = len(time) // 2
    loading_idx = np.arange(0, midpoint)
    unloading_idx = np.arange(midpoint, len(time))

    # Calculate area between loading and unloading curves using trapezoidal integration
    loading_force = fz[loading_idx]
    loading_pos = pz[loading_idx]
    unloading_force = fz[unloading_idx]
    unloading_pos = pz[unloading_idx]

    # Interpolate to common position points to calculate area
    if len(loading_pos) > 2 and len(unloading_pos) > 2:  # Ensure enough points
        common_pos = np.linspace(max(np.min(loading_pos), np.min(unloading_pos)),
                                 min(np.max(loading_pos), np.max(unloading_pos)), 100)

        loading_interp = interp1d(loading_pos, loading_force, bounds_error=False, fill_value=0)
        unloading_interp = interp1d(unloading_pos, unloading_force, bounds_error=False, fill_value=0)

        loading_force_interp = loading_interp(common_pos)
        unloading_force_interp = unloading_interp(common_pos)

        hysteresis_area = np.trapz(np.abs(loading_force_interp - unloading_force_interp), common_pos)
    else:
        hysteresis_area = 0

    # 5. Position Response Rate
    # Calculate the rate of position change during initial loading
    pos_rate = np.diff(pz[:force_max_idx + 1]) / np.diff(time[:force_max_idx + 1])
    max_pos_rate = np.max(pos_rate) if len(pos_rate) > 0 else 0

    # 6. Force Oscillation
    # Calculate standard deviation in force during steady state as a measure of oscillations
    steady_state_region = fz[steady_state_start_idx:steady_state_start_idx + 150]
    force_oscillation = np.std(steady_state_region)

    # 7. Force Relaxation Ratio - Captures material relaxation properties
    if len(fz_touch) > 100:
        early_touch_force = np.mean(fz_touch[10:30])  # Skip the first few points
        late_touch_force = np.mean(fz_touch[-30:])
        force_relaxation = (early_touch_force - late_touch_force) / early_touch_force
    else:
        force_relaxation = 0

    # 8. Initial Stiffness vs Later Stiffness (Stiffness Change)
    # This can capture non-linear elastic behavior
    if len(pz) > 100 and len(fz) > 100:
        early_stiffness_idx = np.where(fz >= 0.3 * force_max)[0][0]
        early_stiffness = fz[early_stiffness_idx] / (pz[early_stiffness_idx] - pz[0])

        late_stiffness_idx = np.where(fz >= 0.7 * force_max)[0][0]
        late_stiffness = (fz[late_stiffness_idx] - fz[early_stiffness_idx]) / (
                    pz[late_stiffness_idx] - pz[early_stiffness_idx])

        stiffness_ratio = late_stiffness / early_stiffness if early_stiffness != 0 else 0
    else:
        stiffness_ratio = 0

    # Return all features including the new ones
    return (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke, fi, pi, time_i, P_ss, offset,
            force_max, time_to_max, force_overshoot, peak_width, hysteresis_area, max_pos_rate,
            force_oscillation, force_relaxation, stiffness_ratio)

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
        if ((posx - center_x) ** 2 + (posy - center_y) ** 2) <= (radius ) ** 2:
            return (i // 5) + 1  # Assign label based on group of 5 circles
    return 0  # Default label


def organize_df(df_input: DataFrame, centers, radii) -> DataFrame | None:
    df_input['forceZ'] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    offset = np.mean(df_input['forceZ'][df_input['isArrived_Festo'] == 1].head(20))
    df_input['forceZ'] = (df_input['forceZ'] - offset)  #/  np.mean(df_input['forceZ'][df_input['isTouching_SMAC'] == 1][ -30:])  # / np.std(df_input['forceZ'][df_input['isArrived_Festo'] == 1].tolist())   #  # /
    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    posx = np.mean(df_input['posx'][df_input['isArrived_Festo'] == 1] + df_input['posx_d'][df_input['isArrived_Festo'] == 1] / 1000)
    posy = np.mean(df_input['posy'][df_input['isArrived_Festo'] == 1] + df_input['posy_d'][df_input['isArrived_Festo'] == 1] / 1000)

    if posx < 20 and posy < 20:
        return None

    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000

    (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke, fi, pi, time_i, P_ss, offset,
     force_max, time_to_max, force_overshoot, peak_width, hysteresis_area, max_pos_rate,
     force_oscillation, force_relaxation, stiffness_ratio) = extract_features(df_input)  # change label

    label = get_label(posx, posy, centers, radii)

    new_df = DataFrame({
        "posx": posx,
        "posy": posy,
        "posz": [df_input['posz'].tolist()],
        "Stiffness": stiffness,
        "Tau": tau,
        "Force Steady State": F_ss,
        "Power": power,
        "Entropy": entropy,
        "Upstroke": upstroke,
        "Downstroke": downstroke,
        "Dominant Frequency": psd_peak,
        "P_ss": P_ss,
        "offset": offset,
        "force_max": force_max,
        "time_to_max": time_to_max,
        "force_overshoot": force_overshoot,
        "force_relaxation": force_relaxation,
        "stiffness_ratio": stiffness_ratio,
        "force_oscillation": force_oscillation,
        "peak_width": peak_width,
        "hysteresis_area": hysteresis_area,
        "max_pos_rate": max_pos_rate,

        # "Fz": [df_input['forceZ'].tolist()],
        "t": [df_input['CPXEts'].tolist()],
        "Fz_s": [df_input['Fz_s'].tolist()],
        "posz_s": [df_input['posz_s'].tolist()],
        # "Touching": [df_input['isTouching_SMAC'].tolist()],
        "Fi": [fi],
        "Pi": [pi],
        "Timei": [time_i],
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
        flag = (df['isTouching_SMAC'] == 0).all()
        if flag: continue
        df_ta = organize_df(df, centers, radii)
        if df_ta is not None and len(df_ta) > 0:
            df_ta.insert(0, "Source", file.split("_")[-1].split(".")[0])
            df_list.append(df_ta)

    return pd.concat(df_list, ignore_index=True)
