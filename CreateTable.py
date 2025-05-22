import glob
import json

import antropy
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
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
    # Force normalization
    offset = np.mean(data['Fz_s'][data['isArrived_Festo'] == 1])
    data['Fz_s'] = (data['Fz_s'] - offset) / np.mean(data['Fz_s'][data['isTouching_SMAC'] == 1][-30:])
    # Position normalization
    offset_p = np.mean(data['posz_s'][data['isArrived_Festo'] == 1])
    data['posz_s'] = (data['posz_s'] - offset_p) / np.mean(data['posz_s'][data['isTouching_SMAC'] == 1][-30:])

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

    # Calculate key indices
    force_max = np.max(fi)
    force_max_idx = np.argmax(fi)
    steady_state_start_idx = min(force_max_idx + int(len(fi) * 0.05), len(fi) - 101)
    steady_state_force = np.mean(
        fi[steady_state_start_idx:steady_state_start_idx + 100]) if steady_state_start_idx < len(fi) - 100 else np.mean(
        fi[-100:])

    # Calculate force thresholds
    low_threshold = 0.1 * force_max
    mid_threshold = 0.5 * force_max
    high_threshold = 0.75 * force_max

    # Force Decay Rate (τ)
    decay_time = time_t - time_t[0]
    decay_force = np.log(np.maximum(ft, 1e-10))  # Avoid log(0)
    tau_result = linregress(decay_time, decay_force)
    tau = -1 / tau_result.slope if tau_result.slope != 0 else None

    # Steady-State Force
    F_ss = steady_state_force
    P_ss = np.mean(pt[-30:]) if len(pt) >= 30 else None

    # Calculate stiffness using pre-computed thresholds
    stiffness = None
    upstroke = None
    downstroke = None

    # Find indices efficiently
    low_idx_up = np.where(fi >= low_threshold)[0]
    mid_idx_up = np.where(fi >= mid_threshold)[0]

    post_peak_indices = np.where((fi <= high_threshold) & (np.arange(len(fi)) > force_max_idx))[0]
    down_end_indices = np.where((fi <= low_threshold) & (np.arange(len(fi)) > force_max_idx))[0]

    if len(low_idx_up) > 0:
        up_start_idx = low_idx_up[0]

        # Calculate stiffness
        if pi[force_max_idx] != pi[up_start_idx]:
            stiffness = (force_max - fi[up_start_idx]) / (pi[force_max_idx] - pi[up_start_idx])

        # Calculate upstroke
        if len(mid_idx_up) > 0:
            mid_up_idx = mid_idx_up[0]
            if pi[mid_up_idx] != pi[up_start_idx]:
                upstroke = (fi[mid_up_idx] - fi[up_start_idx]) / (pi[mid_up_idx] - pi[up_start_idx])

        # Calculate downstrokes
        if len(post_peak_indices) > 0:
            down1_idx = post_peak_indices[0]
            if len(down_end_indices) > 0:
                down_end_idx = down_end_indices[0]
                if pi[down1_idx] != pi[down_end_idx]:
                    downstroke = (fi[down1_idx] - fi[down_end_idx]) / (pi[down1_idx] - pi[down_end_idx])

    # Entropy
    hist, _ = np.histogram(ft, bins=22, density=True)
    hist_positive = hist[hist > 0]
    entropy = -np.sum(hist_positive * np.log2(hist_positive)) if len(hist_positive) > 0 else None

    # Force overshoot
    force_overshoot = (force_max - F_ss) / F_ss if F_ss > 0 else 0

    # Force relaxation
    if len(ft) > 100:
        force_relaxation = (force_max - np.mean(ft[-50:])) / force_max if force_max > 0 else 0
    else:
        force_relaxation = 0

    # Force/position RMS
    force_rms = np.sqrt(np.mean(np.square(fi)))

    # Damping coefficient (optimized to avoid singular matrix errors)
    try:
        # Calculate velocity
        velocity = np.gradient(pi, time_i)
        A = np.vstack([pi, velocity]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, fi, rcond=1e-10)
        _, damping_coefficient = coeffs
    except:
        damping_coefficient = None

    # Contact area estimation
    contact_area = None
    if stiffness and stiffness > 0:
        R = 0.005  # Estimated radius in meters
        nu = 0.5  # Typical Poisson's ratio
        contact_area = - np.pi * ((3 * (1 - nu * 2) * force_max * R / (4 * stiffness)) * (
                2 / 3))  # Minus is to ensure right format for smoothing during classification

    # Force peak-to-peak
    force_ptp = np.ptp(fi)

    # Hjorth parameters
    activity = np.var(fi)
    mobility = np.sqrt(np.var(np.diff(fi)) / activity) if activity > 0 else None

    # Teager-Kaiser Energy Operator
    tkeo_mean = None
    try:
        tkeo = fi[1:-1] ** 2 - fi[:-2] * fi[2:]
        tkeo_mean = np.mean(tkeo)
    except:
        pass

    # Correlation between force and position
    correlation_fp = np.corrcoef(fi, pi)[0, 1] if len(fi) > 1 and len(pi) > 1 else None

    # Calculate peak ratio
    peak_ratio = force_max / np.max(pi) if np.max(pi) > 0 else None

    # Ensure signal has sufficient variance for meaningful analysis
    if np.std(fi) > 1e-6:
        try:
            # Sample_entropy
            sample_entropy = antropy.sample_entropy(fi)
        except Exception:
            sample_entropy = np.nan
    else:
        sample_entropy = np.nan

    # Return all features in tuple format
    return (stiffness, tau, entropy, upstroke, downstroke, fi, pi, time_i, P_ss,
            force_overshoot, force_relaxation,
            force_rms, damping_coefficient, contact_area, force_ptp,
            activity, mobility,
            tkeo_mean, correlation_fp, peak_ratio, sample_entropy)


def extract_curve_features(position, force):
    features = {}
    peak_idx = np.argmax(force)

    # Loading skewness and kurtosis
    if peak_idx > 5:
        features['loading_skewness'] = skew(force[:peak_idx + 1])
        features['loading_kurtosis'] = kurtosis(force[:peak_idx + 1])
    else:
        features['loading_skewness'] = np.nan
        features['loading_kurtosis'] = np.nan

    # Polynomial curve fits
    if peak_idx > 15:
        x_load = position[:peak_idx + 1]
        y_load = force[:peak_idx + 1]

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
        features['cubic_coefficient'] = np.nan
        features['quartic_coefficient'] = np.nan

    return features


def compute_hysteresis_features_for_df(df, force_column='Fi', pos_column='Pi', threshold=0.1):
    """
    Given a DataFrame (with force and position data stored as arrays/lists in each row),
    compute the hysteresis features using extract_curve_features() and add them as new columns.
    """
    # List of hysteresis features
    hysteresis_features = ['loading_skewness', 'loading_kurtosis', 'cubic_coefficient', 'quartic_coefficient']

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

    # Shift, rotate, and shift back *without rounding*
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
    """
    Determine the material label for a point (posx, posy) by checking
    which inclusion circle it falls into. Returns 0 if none match.
    """
    for center, radius, material in zip(centers, radii, materials):
        center_x, center_y = center
        # Check if point is within the circle using proper squared-distance
        if (posx - center_x) ** 2 + (posy - center_y) ** 2 <= radius ** 2:
            return materials_label[material]
    return 0  # Default label if point lies outside all circles


def organize_df(df_input: DataFrame, centers, radii, materials) -> DataFrame | None:
    df_input['forceZ'] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000
    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    posx = np.round(np.mean(df_input['posx'][df_input['isArrived_Festo'] == 1] + df_input['posx_d'][
        df_input['isArrived_Festo'] == 1] / 1000))
    posy = np.round(np.mean(df_input['posy'][df_input['isArrived_Festo'] == 1] + df_input['posy_d'][
        df_input['isArrived_Festo'] == 1] / 1000))

    if posx < 50 or posy < 50:
        return None

    tuple = extract_features(df_input)
    if tuple is None:
        return None

    (stiffness, tau, entropy, upstroke, downstroke, fi, pi, time_i, P_ss,
     force_overshoot, force_relaxation, force_rms, damping_coefficient, contact_area,
     force_ptp, activity, mobility, tkeo_mean, correlation_fp, peak_ratio, sample_entropy
     ) = tuple

    label = get_label(posx, posy, centers, radii, materials)

    new_df = DataFrame({
        "posx": posx,
        "posy": posy,
        "posz": [df_input['posz'].tolist()],
        "Stiffness": stiffness,
        "Tau": tau,
        "Entropy": entropy,
        "Upstroke": upstroke,
        "Downstroke": downstroke,
        "P_ss": P_ss,
        "force_overshoot": force_overshoot,
        "force_relaxation": force_relaxation,
        "force_rms": force_rms,
        "damping_coefficient": damping_coefficient,
        "contact_area": contact_area,
        "force_ptp": force_ptp,
        "activity": activity,
        "mobility": mobility,
        "tkeo_mean": tkeo_mean,
        "correlation_fp": correlation_fp,
        "peak_ratio": peak_ratio,
        "sample_entropy": sample_entropy,

        # Keep raw signals for later processing
        "t": [df_input['CPXEts'].tolist()],
        "Fz_s": [df_input['Fz_s'].tolist()],
        "posz_s": [df_input['posz_s'].tolist()],
        "Fi": [fi],
        "Pi": [pi],
        "Timei": [time_i],
        "label": label,
    })

    new_df = compute_hysteresis_features_for_df(new_df, force_column='Fi', pos_column='Pi')
    return new_df


def create_df(path: str = 'DamasconeA/Dataset/*.csv', symmetric=False, symm_v=False, angle=0,
              offset=(49, 49)) -> DataFrame | None:
    # Load JSON file
    strings = path.split('/')
    with open(strings[0] + "/" + strings[1] + "/phantom_metadata.json", "r") as file:
        json_data = json.load(file)

    # Create lists of centers + radii (for each labeled area)
    centers, radii, materials = find_inclusions(json_data, symmetric=symmetric, angle=angle, symm_v=symm_v,
                                                offset=offset)

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
    combined_df = pd.concat(non_empty_df_list, ignore_index=True)
    return combined_df
