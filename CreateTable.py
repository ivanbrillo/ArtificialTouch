from sys import dont_write_bytecode

import pandas as pd
import glob
import numpy as np
import json
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis

materials_label = {
    "Dragon Skin shore 20A": 1,
    "Dragon Skin shore 30A": 2,
    "SORTA-Clear shore 40A": 3,
    "PDMS shore 44": 4,
    "Econ shore 80A": 5
}


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

    if len(time) == 0:
        return None

    time_i = np.arange(time[0], time[-1], 0.001)
    time_t = np.arange(time_touch[0], time_touch[-1], 0.001)

    f = interp1d(time, fz, kind='cubic')
    f2 = interp1d(time_touch, fz_touch, kind='cubic')
    fi = f(time_i)
    ft = f2(time_t)

    g = interp1d(time, pz, kind='cubic')
    g2 = interp1d(time_touch, pz_touch, kind='cubic')
    pi = g(time_i)
    pt = g2(time_t)

    # Force Decay Rate (τ)
    decay_time = time_t - time_t[0]
    decay_force = np.log(ft)  # Log transform to get a linear function like ln(F) = -t/tau + c
    tau, _, _, _, _ = linregress(decay_time, decay_force)
    tau = -1 / tau  # Convert to time constant

    # Steady-State Force (F_ss) - Mean force at end of signal
    F_ss = np.mean(ft[-30:])
    P_ss = np.mean(ft[-30:])

    # Ensure fi has values meeting the conditions before accessing indices
    if np.any(fi >= 0.1):
        start_idx = np.where(fi >= 0.1)[0][0]
        stiffness = (np.max(fi) - 0.1) / (pi[np.argmax(fi)] - pi[start_idx])

        if np.any(fi > 0.5):
            idx1 = np.where(fi > 0.5)[0][0]  # First index where force > 0.5
            idx2 = np.where(fi >= 0.1)[0][0]  # First index where force > 0.1
            upstroke = (fi[idx1] - fi[idx2]) / (time_i[idx1] - time_i[idx2])
            idx1 = np.where(fi > 0.5)[0][-1]  # First index where force > 0.5
            idx2 = np.where(fi >= 0.1)[0][-1]  # First index where force > 0.1
            downstroke = (fi[idx1] - fi[idx2]) / (time_i[idx1] - time_i[idx2])
        else:
            upstroke = None
            downstroke = None
    else:
        stiffness = None  # Or set to a default value or handle as needed
        upstroke = None
        downstroke = None

    # Hardness
    # start_idx = np.where(fi >= 0.1)[0][0]
    # stiffness = (np.max(fi) - 0.1) / (pi[np.argmax(fi)] - pi[start_idx])
    # idx1 = np.where(fi > 0.5)[0][0]  # First index where force > 0.5
    # idx2 = np.where(fi > 0.1)[0][0]  # First index where force > 0.1
    # upstroke = (fi[idx1] - fi[idx2]) / (time_i[idx1] - time_i[idx2])
    #
    # idx1 = np.where(fi > 0.5)[0][-1]  # First index where force > 0.5
    # idx2 = np.where(fi > 0.1)[0][-1]  # First index where force > 0.1
    # downstroke = (fi[idx1] - fi[idx2]) / (time_i[idx1] - time_i[idx2])

    offset = np.mean(ft[:5]) - np.mean(ft[:-10])

    # PSD
    freqs, psd = welch(ft, fs=1 / np.median(np.diff(time_i)))
    power = np.mean(np.square(ft)) / (time_i[-1] - time_i[0])

    psd_peak = freqs[np.argmax(psd)]

    # Entropy of the Force Signal
    try:
        prob_dist = np.histogram(ft, bins=30, density=True)[
            0]  # Probability distribution
        prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
        entropy = -np.sum(prob_dist * np.log2(
            prob_dist))  # Shannon entropy => measure of randomness (estimation of fluctuations in the signal)
    except Exception:
        entropy = None

    # NEW FEATURES

    # 1. Initial Force Peak Characteristics
    force_max = np.max(fi)
    force_max_idx = np.argmax(fi)
    time_to_max = time_i[force_max_idx] - time_i[0]

    # 2. Force Overshoot - This seems prominent in your graphs
    steady_state_start_idx = force_max_idx + int(len(fi) * 0.1)  # Skip a bit after peak
    steady_state_force = np.mean(fi[steady_state_start_idx:steady_state_start_idx + 100])
    force_overshoot = (force_max - steady_state_force) / steady_state_force

    # 3. Initial Peak Width - Different materials show different peak widths
    half_height = (force_max - fi[0]) / 2 + fi[0]
    above_half_height = np.where(fi > half_height)[0]

    if len(above_half_height) > 0:
        peak_width = time_i[above_half_height[-1]] - time_i[above_half_height[0]]
    else:
        peak_width = None

    # 5. Position Response Rate
    # Calculate the rate of position change during initial loading
    pos_rate = np.diff(pi[:force_max_idx + 1]) / np.diff(time_i[:force_max_idx + 1])
    max_pos_rate = np.max(pos_rate) if len(pos_rate) > 0 else 0

    # 6. Force Oscillation
    # Calculate standard deviation in force during steady state as a measure of oscillations
    steady_state_region = fi[steady_state_start_idx:steady_state_start_idx + 150]
    force_oscillation = np.std(steady_state_region)

    # 7. Force Relaxation Ratio - Captures material relaxation properties
    if len(ft) > 100:
        early_touch_force = np.mean(ft[10:30])  # Skip the first few points
        late_touch_force = np.mean(ft[-30:])
        force_relaxation = (early_touch_force - late_touch_force) / early_touch_force
    else:
        force_relaxation = 0

    # 8. Initial Stiffness vs Later Stiffness (Stiffness Change)
    # This can capture non-linear elastic behavior
    if len(pi) > 100 and len(fi) > 100:

        if np.any(fi >= 0.3 * force_max):
            early_stiffness_idx = np.where(fi >= 0.3 * force_max)[0][0]
            early_stiffness = fi[early_stiffness_idx] / (pi[early_stiffness_idx] - pi[0])
        else:
            early_stiffness = None

        if np.any(fi >= 0.7 * force_max):
            late_stiffness_idx = np.where(fi >= 0.7 * force_max)[0][0]
            late_stiffness = (fi[late_stiffness_idx] - fi[early_stiffness_idx]) / (
                    pi[late_stiffness_idx] - pi[early_stiffness_idx])
        else:
            late_stiffness = None

        stiffness_ratio = late_stiffness / early_stiffness if (
                early_stiffness is not None and early_stiffness != 0 and late_stiffness is not None) else None
    else:
        stiffness_ratio = None

    # Return all features including the new ones
    return (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke, fi, pi, time_i, P_ss, offset,
            force_max, time_to_max, force_overshoot, peak_width, max_pos_rate,
            force_oscillation, force_relaxation, stiffness_ratio)


def extract_curve_features(position, force):
    features = {}

    # --- 1. Peak detection ---
    peak_idx = np.argmax(force)
    features['peak_position'] = position[peak_idx]

    # --- 2. Hysteresis area ---
    # Ensure the curve is closed (append first point if necessary)
    if not np.array_equal(position[0], position[-1]) or not np.array_equal(force[0], force[-1]):
        position_closed = np.append(position, position[0])
        force_closed = np.append(force, force[0])
    else:
        position_closed = position
        force_closed = force
    features['hysteresis_area'] = 0.5 * abs(np.sum(
        position_closed[:-1] * force_closed[1:] - position_closed[1:] * force_closed[:-1]
    ))

    # --- 3. Loading energy (area under loading curve) ---
    features['loading_energy'] = np.trapz(force[:peak_idx + 1], position[:peak_idx + 1])

    # --- 4. Loading nonlinearity ---
    if peak_idx > 10:
        try:
            quad_fit = np.polyfit(position[:peak_idx + 1], force[:peak_idx + 1], 2)
            features['loading_nonlinearity'] = abs(quad_fit[0])
        except Exception:
            features['loading_nonlinearity'] = np.nan
    else:
        features['loading_nonlinearity'] = np.nan

    # --- 5. Loading-to-unloading area ratio ---
    if peak_idx > 10 and peak_idx < len(position) - 10:
        loading_area = np.trapz(force[:peak_idx + 1], position[:peak_idx + 1])
        unloading_area = np.trapz(force[peak_idx:], position[peak_idx:])
        features['loading_unloading_area_ratio'] = loading_area / abs(unloading_area) if unloading_area != 0 else np.nan
    else:
        features['loading_unloading_area_ratio'] = np.nan

    # --- 6. Force ratio (75% displacement / 25% displacement) ---
    if peak_idx > 5:
        peak_pos = position[peak_idx]
        idx_25 = np.argmin(np.abs(position[:peak_idx + 1] - 0.25 * peak_pos))
        idx_75 = np.argmin(np.abs(position[:peak_idx + 1] - 0.75 * peak_pos))
        f25 = force[idx_25]
        f75 = force[idx_75]
        features['force_ratio_75_25'] = f75 / f25 if f25 != 0 else np.nan
    else:
        features['force_ratio_75_25'] = np.nan

    # --- 7. Loading skewness and kurtosis ---
    if peak_idx > 5:
        features['loading_skewness'] = skew(force[:peak_idx + 1])
        features['loading_kurtosis'] = kurtosis(force[:peak_idx + 1])
    else:
        features['loading_skewness'] = np.nan
        features['loading_kurtosis'] = np.nan

    # --- 8. Polynomial curve fits ---
    if peak_idx > 15:
        x_load = position[:peak_idx + 1]
        y_load = force[:peak_idx + 1]
        x_norm = (x_load - np.min(x_load)) / (np.ptp(x_load) if np.ptp(x_load) != 0 else 1)
        y_norm = y_load / np.max(y_load) if np.max(y_load) != 0 else y_load

        for degree in [3, 4, 5]:
            try:
                poly_fit = np.polyfit(x_norm, y_norm, degree)
                for i, coef in enumerate(poly_fit):
                    features[f'poly{degree}_coef{i}'] = coef
            except Exception:
                for i in range(degree + 1):
                    features[f'poly{degree}_coef{i}'] = np.nan

        try:
            cubic_fit = np.polyfit(x_load, y_load, 3)
            features['cubic_coefficient'] = cubic_fit[0]
        except Exception:
            features['cubic_coefficient'] = np.nan
        try:
            quartic_fit = np.polyfit(x_load, y_load, 4)
            features['quartic_coefficient'] = quartic_fit[0]
        except Exception:
            features['quartic_coefficient'] = np.nan
    else:
        for degree in [3, 4, 5]:
            for i in range(degree + 1):
                features[f'poly{degree}_coef{i}'] = np.nan
        features['cubic_coefficient'] = np.nan
        features['quartic_coefficient'] = np.nan

    # --- 9. Segmentation features (only segments 2 and 3) ---
    if peak_idx > 15:
        num_segments = 4
        segment_size = (peak_idx + 1) // num_segments
        for seg in [2, 3]:
            start = seg * segment_size
            end = (seg + 1) * segment_size if seg < num_segments - 1 else peak_idx + 1
            if end - start > 5:
                seg_pos = position[start:end]
                seg_force = force[start:end]
                slope = (seg_force[-1] - seg_force[0]) / (seg_pos[-1] - seg_pos[0]) if (seg_pos[-1] - seg_pos[
                    0]) != 0 else np.nan
                std = np.std(seg_force)
                seg_skew = skew(seg_force) if len(seg_force) > 3 else np.nan
            else:
                slope = np.nan
                std = np.nan
                seg_skew = np.nan
            features[f'segment{seg}_slope'] = slope
            features[f'segment{seg}_force_std'] = std
            features[f'segment{seg}_skew'] = seg_skew
    else:
        for seg in [2, 3]:
            features[f'segment{seg}_slope'] = np.nan
            features[f'segment{seg}_force_std'] = np.nan
            features[f'segment{seg}_skew'] = np.nan

    return features


def compute_hysteresis_features_for_df(df, force_column='Fi', pos_column='Pi', threshold=0.1):
    """
    Given a DataFrame (with force and position data stored as arrays/lists in each row),
    compute the hysteresis features using extract_curve_features() and add them as new columns.
    """
    # List of hysteresis features (update if additional features are added in extract_curve_features)
    hysteresis_features = [
        'peak_position', 'hysteresis_area', 'loading_energy', 'loading_nonlinearity',
        'loading_unloading_area_ratio', 'force_ratio_75_25', 'loading_skewness',
        'loading_kurtosis', 'poly3_coef0', 'poly3_coef1', 'poly3_coef2', 'poly3_coef3',
        'poly4_coef0', 'poly4_coef1', 'poly4_coef2', 'poly4_coef3', 'poly4_coef4',
        'poly5_coef0', 'poly5_coef1', 'poly5_coef2', 'poly5_coef3', 'poly5_coef4',
        'segment2_slope', 'segment3_slope', 'segment2_force_std', 'segment3_force_std',
        'segment2_skew', 'segment3_skew', 'cubic_coefficient', 'quartic_coefficient'
    ]

    # Initialize the hysteresis feature columns if they don't exist
    for feat in hysteresis_features:
        if feat not in df.columns:
            df[feat] = np.nan

    # Loop through each row and compute the features
    for idx, row in df.iterrows():
        force = np.array(row[force_column])
        position = np.array(row[pos_column])
        # Align position so that the first force value above the threshold is at zero
        force_above_threshold = force > threshold
        if np.any(force_above_threshold):

            start_idx = np.where(force_above_threshold)[0][0]
            aligned_position = position - position[start_idx]
            feats = extract_curve_features(aligned_position, force)
            for key, value in feats.items():
                df.at[idx, key] = value
    return df


import numpy as np


def find_inclusions(json_data, symmetric=False, symm_v=False, angle=0, offset=(49, 49)) -> tuple:
    circles = json_data["Inclusions"]

    # Define small clockwise rotation angle (-0.46 degrees)
    theta = np.radians(-0.46 + angle)  # Negative for clockwise rotation

    # Rotation matrix
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])

    # Rotation center
    center = np.array([100, 100])

    # Extract original centers
    c = np.array([(circle["Position"][0] + offset[0], circle["Position"][1] + offset[1]) for circle in circles])

    # Shift, rotate, and shift back **without rounding**
    c_shifted = c - center

    if symmetric:
        c_shifted = np.array([(-y, -x) for x, y in c_shifted])

    if symm_v:
        c_shifted = np.array([(-x, y) for x, y in c_shifted])


    c_rotated = np.dot(c_shifted, R)
    c_final = c_rotated + center + np.array([-1, -1])  # Keep as float

    # Convert back to list of tuples
    c_final_list = [tuple(point) for point in c_final]

    r = [circle["Diameter"] / 2 for circle in circles]
    materials = [circle["Material"] for circle in circles]

    return c_final_list, r, materials


def get_label(posx, posy, centers, radii, materials) -> int:
    for i in range(len(centers)):
        center_x, center_y = centers[i]
        radius = radii[i]
        material = materials[i]
        # Check if point is within the circle
        if ((posx - center_x) ** 2 + (posy - center_y) ** 2) <= (radius) ** 2:
            return materials_label[material]
            # return (i // 5) + 1  # Assign label based on group of 5 circles
    return 0  # Default label


def organize_df(df_input: DataFrame, centers, radii, materials) -> DataFrame | None:
    df_input['forceZ'] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    offset = np.mean(df_input['forceZ'][df_input['isArrived_Festo'] == 1].head(20))
    df_input['forceZ'] = (df_input['forceZ'] - offset)  # / np.mean(df_input['forceZ'][df_input['isTouching_SMAC'] == 1][ -30:])  # / np.std(df_input['forceZ'][df_input['isArrived_Festo'] == 1].tolist())   #  # /
    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    posx = np.round(np.mean(df_input['posx'][df_input['isArrived_Festo'] == 1] + df_input['posx_d'][
        df_input['isArrived_Festo'] == 1] / 1000))
    posy = np.round(np.mean(df_input['posy'][df_input['isArrived_Festo'] == 1] + df_input['posy_d'][
        df_input['isArrived_Festo'] == 1] / 1000))

    if posx < 50 or posy < 50:
        return None

    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000

    tuple = extract_features(df_input)
    if tuple is None:
        return None

    (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke, fi, pi, time_i, P_ss, offset,
     force_max, time_to_max, force_overshoot, peak_width, max_pos_rate,
     force_oscillation, force_relaxation, stiffness_ratio) = tuple

    label = get_label(posx, posy, centers, radii, materials)

    # row = 'Test' if 110 <= posy <= 125 else 'Train'

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
        "max_pos_rate": max_pos_rate,

        # "Fz": [df_input['forceZ'].tolist()],
        "t": [df_input['CPXEts'].tolist()],
        "Fz_s": [df_input['Fz_s'].tolist()],
        "posz_s": [df_input['posz_s'].tolist()],
        # "Touching": [df_input['isTouching_SMAC'].tolist()],
        "Fi": [fi],
        "Pi": [pi],
        "Timei": [time_i],
        "label": label,
        # "Row": row
    })

    new_df = compute_hysteresis_features_for_df(new_df, force_column='Fi', pos_column='Pi')
    return new_df


def create_df(path: str = 'DamasconeA/Dataset/*.csv', symmetric=False, symm_v=False, angle=0, offset = (49, 49)) -> DataFrame | None:
    # Load JSON file

    strings = path.split('/')
    with open(strings[0] + "/phantom_metadata.json", "r") as file:
        json_data = json.load(file)

    # Create lists of centers + radii (for each labeled area)
    centers, radii, materials = find_inclusions(json_data, symmetric=symmetric, angle=angle, symm_v=symm_v, offset=offset)

    # Get list of all CSV files
    csv_files = glob.glob(path)

    df_list = list()

    for file in csv_files:
        df = pd.read_csv(file)
        flag = (df['isTouching_SMAC'] == 0).all()
        if flag: continue
        df_ta = organize_df(df, centers, radii, materials)
        if df_ta is not None and len(df_ta) > 0:
            df_ta.insert(0, "Source", file.split("_")[-1].split(".")[0])
            df_list.append(df_ta)
        else:
            print(f"File {file} has no valid data after processing.")

    cleaned_df_list = [df.dropna(how='all') for df in df_list]
    non_empty_df_list = [df for df in cleaned_df_list if len(df) > 0]
    return pd.concat(non_empty_df_list, ignore_index=True)
