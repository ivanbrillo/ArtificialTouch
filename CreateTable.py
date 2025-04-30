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
from scipy.fft import fft, fftfreq
from scipy.interpolate import griddata
from scipy.signal import coherence
from skimage.feature import local_binary_pattern
from sklearn.neighbors import LocalOutlierFactor

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

    # Calculate stiffness and stroke rates using thresholds relative to peak force
    if len(fi) > 0:  # Make sure we have data
        peak_force = np.max(fi)
        peak_idx = np.argmax(fi)

        # Define thresholds as percentages of peak force
        low_threshold = 0.1 * peak_force
        mid_threshold = 0.5 * peak_force
        high_threshold = 0.75 * peak_force

        # Check if we have enough force data to analyze
        if peak_force > 0:
            # Find indices for upstroke (start to peak)
            up_start_indices = np.where(fi >= low_threshold)[0]
            if len(up_start_indices) > 0:
                up_start_idx = up_start_indices[0]  # First point exceeding low threshold

                # Calculate overall stiffness from low threshold to peak
                stiffness = (peak_force - fi[up_start_idx]) / (pi[peak_idx] - pi[up_start_idx])

                # Calculate upstroke rate (from low to mid threshold)
                mid_up_indices = np.where(fi >= mid_threshold)[0]
                if len(mid_up_indices) > 0:
                    mid_up_idx = mid_up_indices[0]  # First point exceeding mid threshold
                    upstroke = (fi[mid_up_idx] - fi[up_start_idx]) / (pi[mid_up_idx] - pi[up_start_idx])
                else:
                    upstroke = None

                # Calculate downstroke1 rate (from peak to high threshold on descent)
                post_peak_indices = np.where((fi <= high_threshold) & (np.arange(len(fi)) > peak_idx))[0]
                if len(post_peak_indices) > 0:
                    down1_idx = post_peak_indices[0]  # First point after peak dropping below high threshold
                    downstroke1 = (peak_force - fi[down1_idx]) / (pi[peak_idx] - pi[down1_idx]) # to keep it positive
                else:
                    downstroke1 = None

                # Calculate downstroke2 rate (from high to low threshold on descent)
                down_end_indices = np.where((fi <= low_threshold) & (np.arange(len(fi)) > peak_idx))[0]
                if len(post_peak_indices) > 0 and len(down_end_indices) > 0:
                    down_end_idx = down_end_indices[0]  # First point after peak dropping below low threshold
                    downstroke2 = (fi[down1_idx] - fi[down_end_idx]) / (pi[down1_idx] - pi[down_end_idx]) # to keep it positive
                else:
                    downstroke2 = None
            else:
                stiffness = None
                upstroke = None
                downstroke1 = None
                downstroke2 = None
        else:
            stiffness = None
            upstroke = None
            downstroke1 = None
            downstroke2 = None
    else:
        stiffness = None
        upstroke = None
        downstroke1 = None
        downstroke2 = None

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

    # 2. Force Overshoot
    steady_state_start_idx = force_max_idx + int(len(fi) * 0.05)  # Skip a bit after peak
    steady_state_force = np.mean(fi[steady_state_start_idx:steady_state_start_idx + 100])
    force_overshoot = (force_max - steady_state_force) / steady_state_force

    # 3. Initial Peak Width
    f3db = force_max/ np.sqrt(2)
    above_3db = np.where(fi > f3db)[0]

    if len(above_3db) > 0:
        peak_width = time_i[above_3db[-1]] - time_i[above_3db[0]]
    else:
        peak_width = None

    # 5. Position Response Rate
    # Calculate the rate of position change during initial loading
    pos_rate = np.diff(pi[:force_max_idx + 1]) / np.diff(time_i[:force_max_idx + 1])
    max_pos_rate = np.max(pos_rate) if len(pos_rate) > 0 else 0

    # 6. Force Oscillation
    # Calculate standard deviation in force during steady state as a measure of oscillations
    steady_state_region = fi[steady_state_start_idx:steady_state_start_idx + 100]
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

        if np.any(fi >= 0.5 * force_max):
            early_stiffness_idx = np.where(fi >= 0.5 * force_max)[0][0]
            start_idx = np.where(fi >= 0.1 * force_max)[0][0]
            early_stiffness = (fi[early_stiffness_idx] - fi[start_idx]) / (pi[early_stiffness_idx] - pi[start_idx])
        else:
            early_stiffness = None

        if np.any(fi >= 0.8 * force_max):
            late_stiffness_idx = np.where(fi >= 0.8 * force_max)[0][0]
            late_stiffness = (force_max - fi[early_stiffness_idx]) / (pi[force_max_idx] - pi[late_stiffness_idx])
        else:
            late_stiffness = None

        stiffness_ratio = late_stiffness / early_stiffness if (early_stiffness is not None and early_stiffness != 0 and late_stiffness is not None) else None
    else:
        stiffness_ratio = None

    # Compute velocity and acceleration for jerk
    velocity = np.gradient(pi, time_i)
    acceleration = np.gradient(velocity, time_i)
    jerk = np.gradient(acceleration, time_i)
    jerk_max = np.max(np.abs(jerk))

    # Force/position RMS
    force_rms = np.sqrt(np.mean(np.square(fi)))
    position_rms = np.sqrt(np.mean(np.square(pi)))

    # Zero-crossings
    zero_crossings_force = np.sum(np.diff(np.signbit(fi - np.mean(fi))))
    zero_crossings_position = np.sum(np.diff(np.signbit(pi - np.mean(pi))))

    # Damping coefficient (Kelvin-Voigt model)
    try:
        A = np.vstack([pi, velocity]).T
        k, c = np.linalg.lstsq(A, fi, rcond=None)[0]
        damping_coefficient = c
        elastic_coefficient = k
    except:
        damping_coefficient = None

    # Spectral features from FFT
    n = len(fi)
    fi_fft = fft(fi)
    pi_fft = fft(pi)
    dt = np.mean(np.diff(time_i))
    freq = fftfreq(n, dt)[:n // 2]
    fi_mag = np.abs(fi_fft)[:n // 2]
    pi_mag = np.abs(pi_fft)[:n // 2]

    # Normalize spectra
    fi_mag_norm = fi_mag / np.sum(fi_mag) if np.sum(fi_mag) > 0 else fi_mag
    pi_mag_norm = pi_mag / np.sum(pi_mag) if np.sum(pi_mag) > 0 else pi_mag

    # Spectral centroids
    spectral_centroid_force = np.sum(freq * fi_mag_norm) if np.sum(fi_mag_norm) > 0 else 0
    spectral_centroid_position = np.sum(freq * pi_mag_norm) if np.sum(pi_mag_norm) > 0 else 0

    # Spectral entropy
    fi_mag_norm_pos = fi_mag_norm[fi_mag_norm > 0]
    pi_mag_norm_pos = pi_mag_norm[pi_mag_norm > 0]
    spectral_entropy_force = -np.sum(fi_mag_norm_pos * np.log2(fi_mag_norm_pos)) if len(fi_mag_norm_pos) > 0 else 0
    spectral_entropy_position = -np.sum(pi_mag_norm_pos * np.log2(pi_mag_norm_pos)) if len(pi_mag_norm_pos) > 0 else 0

    # Impedance ratio
    low_freq_idx = (freq < 10) & (freq > 0)
    if np.any(low_freq_idx) and np.any(pi_mag[low_freq_idx] > 0):
        impedance_ratio_lowfreq = np.mean(fi_mag[low_freq_idx] / pi_mag[low_freq_idx])
    else:
        impedance_ratio_lowfreq = None

    # Spectral coherence
    if len(fi) > 0 and len(pi) > 0:
        fs = 1.0 / np.mean(dt)
        f, Cxy = coherence(fi, pi, fs=fs, nperseg=min(256, len(fi) // 4))
        low_freq_idx = f < 10
        high_freq_idx = f > 50
        coherence_LF = np.mean(Cxy[low_freq_idx])
        coherence_HF = np.mean(Cxy[high_freq_idx])
    else:
        coherence_LF = None
        coherence_HF = None

    # Return all features
    return (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke1,downstroke2, fi, pi, time_i, P_ss, offset,
            force_max, time_to_max, force_overshoot, peak_width, max_pos_rate,
            force_oscillation, force_relaxation, stiffness_ratio,
            jerk_max, zero_crossings_force, zero_crossings_position,
            force_rms, position_rms, damping_coefficient,
            spectral_centroid_force, spectral_entropy_force,
            spectral_centroid_position, spectral_entropy_position,
            impedance_ratio_lowfreq, coherence_LF, coherence_HF, elastic_coefficient)

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

def compute_spatial_and_surface_features(df):
    """
    Combined function to compute spatial, surface and LBP features across the 2D grid with interpolation
    for missing points but without adding features for those missing points.
    """

    # Calculate grid dimensions from data
    minx = int(df['posx'].min())
    miny = int(df['posy'].min())
    maxx = int(df['posx'].max())
    maxy = int(df['posy'].max())

    # Adjust grid size based on data boundaries
    width = maxx - minx + 1
    height = maxy - miny + 1

    # Get the set of points that actually exist in the data
    existing_points = set(zip(df['posx'].astype(int), df['posy'].astype(int)))

    # Create grids for spatial analysis with interpolation for calculations
    grids = {}
    for feature in ['Stiffness', 'Upstroke', 'Downstroke1','Downstroke2','Tau','time_to_max']:
        # Create initial grid with NaN values
        grid = np.full((height, width), np.nan)

        # Fill with available data
        for idx, row in df.iterrows():
            x, y = int(row['posx']) - minx, int(row['posy']) - miny
            if 0 <= x < width and 0 <= y < height:
                grid[y, x] = row[feature]

        # Interpolate missing values for calculation purposes
        filled_grid = interpolate_missing_values(grid)
        grids[feature] = filled_grid

    # Create position grid for surface features
    position_grid = np.full((height, width), np.nan)
    for _, row in df.iterrows():
        x, y = int(row['posx']) - minx, int(row['posy']) - miny
        if 0 <= x < width and 0 <= y < height:
            pi = np.array(row['Pi'])
            if len(pi) > 0:
                position_grid[y, x] = pi[-20]

    # Interpolate missing values in position grid
    filled_position_grid = interpolate_missing_values(position_grid)

    # Prepare depth map for LBP
    Stiffness_LBP_map = grids['Stiffness']

    # Define LBP parameters
    lbp_params = [
        (8, 1, 'uniform'),  # P=8, R=1
        (16, 2, 'uniform')  # P=16, R=2
    ]

    # Check if the tau map is valid and normalize it
    if not np.isnan(Stiffness_LBP_map).all():
        # Normalize tau map to [0, 1] for LBP
        if np.ptp(Stiffness_LBP_map) > 0:
            norm_tau_map = (Stiffness_LBP_map - np.min(Stiffness_LBP_map)) / np.ptp(Stiffness_LBP_map)
        else:
            norm_tau_map = Stiffness_LBP_map - np.min(Stiffness_LBP_map)

        # Compute LBP maps
        lbp_maps = {}
        for P, R, method in lbp_params:
            try:
                lbp_maps[(P, R)] = local_binary_pattern(norm_tau_map, P, R, method=method)
            except Exception:
                lbp_maps[(P, R)] = None
    else:
        lbp_maps = None

    # Calculate global statistics (for deviation features)
    global_mean_stiffness = np.nanmean(grids['Stiffness'])
    global_mean_position = np.nanmean(filled_position_grid)
    global_std_stiffness = np.nanstd(grids['Stiffness'])
    global_std_position = np.nanstd(filled_position_grid)
    global_median_stiffness = np.nanmedian(grids['Stiffness'])
    global_median_position = np.nanmedian(filled_position_grid)


    # Only process for EXISTING points
    for idx, row in df.iterrows():
        orig_x, orig_y = int(row['posx']), int(row['posy'])

        # Check if this point actually exists in the original data
        if (orig_x, orig_y) in existing_points:
            x, y = orig_x - minx, orig_y - miny

            # Only process if the point is within our grid
            if 0 <= x < width and 0 <= y < height:
                # Spatial features
                # Define 3x3 patch
                x_min, x_max = max(0, x - 1), min(width - 1, x + 1) + 1
                y_min, y_max = max(0, y - 1), min(height - 1, y + 1) + 1

                # Extract patches
                for feature in ['Stiffness', 'Upstroke', 'Downstroke1','Downstroke2','Tau','time_to_max']:
                    patch = grids[feature][y_min:y_max, x_min:x_max].flatten()
                    patch = patch[~np.isnan(patch)]
                    if len(patch) > 0:
                        # Local mean
                        df.at[idx, f'local_mean_{feature}'] = np.mean(patch)

                        # Local std
                        df.at[idx, f'local_std_{feature}'] = np.std(patch)

                        # Local skewness
                        df.at[idx, f'local_skewness_{feature}'] = skew(patch)

                        # Local kurtosis
                        df.at[idx, f'local_kurtosis_{feature}'] = kurtosis(patch)

                        # Local range (max - min)
                        df.at[idx, f'local_range_{feature}'] = np.ptp(patch)

                # Gradient norm - use central differences where possible
                if 0 < x < width - 1:
                    gx = (grids['Stiffness'][y, x + 1] - grids['Stiffness'][y, x - 1]) / 2
                elif x == 0:
                    gx = grids['Stiffness'][y, x + 1] - grids['Stiffness'][y, x]
                else:  # x == width - 1
                    gx = grids['Stiffness'][y, x] - grids['Stiffness'][y, x - 1]

                if 0 < y < height - 1:
                    gy = (grids['Stiffness'][y + 1, x] - grids['Stiffness'][y - 1, x]) / 2
                elif y == 0:
                    gy = grids['Stiffness'][y + 1, x] - grids['Stiffness'][y, x]
                else:  # y == height - 1
                    gy = grids['Stiffness'][y, x] - grids['Stiffness'][y - 1, x]

                df.at[idx, 'local_gradient_norm_stiffness'] = np.sqrt(gx ** 2 + gy ** 2)

                # Deviation from global mean
                df.at[idx, 'stiffness_deviation_from_global'] = grids['Stiffness'][y, x] - global_mean_stiffness
                df.at[idx, 'position_deviation_from_global'] = filled_position_grid[y, x] - global_mean_position
                # Deviation from global median
                df.at[idx, 'stiffness_deviation_from_global_median'] = grids['Stiffness'][y, x] - global_median_stiffness
                df.at[idx, 'position_deviation_from_global_median'] = filled_position_grid[y, x] - global_median_position
                # Z-score relative to global distribution
                df.at[idx, 'stiffness_z_score'] = (grids['Stiffness'][y, x] - global_mean_stiffness) / global_std_stiffness
                df.at[idx, 'position_z_score'] = (filled_position_grid[y, x] - global_mean_position) / global_std_position

                # LBP Features
                for P, R, method in lbp_params:
                    if lbp_maps is None: continue
                    lbp_map = lbp_maps.get((P, R))
                    if lbp_map is None: continue

                    # LBP code at (x, y)
                    df.at[idx, f'lbp_code_P{P}_R{R}'] = lbp_map[y, x]

                    # Local neighborhood (5x5)
                    y_min, y_max = max(0, y - 2), min(height - 1, y + 2) + 1
                    x_min, x_max = max(0, x - 2), min(width - 1, x + 2) + 1
                    local_lbp = lbp_map[y_min:y_max, x_min:x_max].flatten()

                    # Histogram bin count
                    if method == 'uniform' and P == 8:
                        n_bins = 59
                    elif method == 'uniform' and P == 16:
                        n_bins = 243
                    else:
                        n_bins = P + 2

                    hist, _ = np.histogram(local_lbp, bins=n_bins, range=(0, n_bins), density=True)

                    # Top LBP bins
                    top_bins = np.argsort(hist)[-3:]
                    for i, bin_idx in enumerate(top_bins):
                        df.at[idx, f'lbp_bin{i + 1}_P{P}_R{R}'] = bin_idx
                        df.at[idx, f'lbp_val{i + 1}_P{P}_R{R}'] = hist[bin_idx]

                    # Statistical features
                    df.at[idx, f'lbp_mean_P{P}_R{R}'] = np.mean(local_lbp)
                    df.at[idx, f'lbp_std_P{P}_R{R}'] = np.std(local_lbp)
                    df.at[idx, f'lbp_entropy_P{P}_R{R}'] = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

                # LOCAL SURFACE METRICS - multiple window sizes (USING STIFFNESS)
                window_sizes = [3, 5, 7]  # 3x3, 5x5, 7x7 windows

                for window_size in window_sizes:
                    half_window = window_size // 2
                    # Define neighborhood for surface metrics
                    y_min_surf = max(0, y - half_window)
                    y_max_surf = min(height - 1, y + half_window) + 1
                    x_min_surf = max(0, x - half_window)
                    x_max_surf = min(width - 1, x + half_window) + 1

                    # Extract neighborhood from filled position grid
                    local_positions = filled_position_grid[y_min_surf:y_max_surf, x_min_surf:x_max_surf].flatten()
                    local_positions = local_positions[~np.isnan(local_positions)]
                    # Extract neighborhood data for multiple features
                    local_stiffness = grids['Stiffness'][y_min_surf:y_max_surf, x_min_surf:x_max_surf].flatten()
                    # Remove NaN values
                    valid_indices = ~(np.isnan(local_stiffness) | np.isnan(local_positions))
                    local_stiffness = local_stiffness[valid_indices]

                    if len(local_positions) > 0:
                        # Surface roughness
                        local_mean = np.mean(local_positions)
                        df.at[idx, f'local_Ra_w{window_size}'] = np.mean(np.abs(local_positions - local_mean))
                        df.at[idx, f'local_Rq_w{window_size}'] = np.sqrt(np.mean(np.square(local_positions - local_mean)))

                        # Surface statistical properties
                        if len(local_positions) > 2:  # Need at least 3 points for skew and kurtosis
                            df.at[idx, f'local_skewness_w{window_size}'] = skew(local_positions)
                            df.at[idx, f'local_kurtosis_w{window_size}'] = kurtosis(local_positions)

                        # Calculate local max peak height and valley depth
                        if len(local_positions) > 1:
                            df.at[idx, f'local_peak_height_w{window_size}'] = np.max(local_positions) - local_mean
                            df.at[idx, f'local_valley_depth_w{window_size}'] = local_mean - np.min(local_positions)


                    # Local anomaly scores
                    # Simple anomaly detection - distance from local mean in standard deviations
                    local_mean_stiff = np.mean(local_stiffness)
                    df.at[idx, f'local_stiffness_w{window_size}'] = local_mean_stiff
                    local_std_stiff = np.std(local_stiffness)
                    if local_std_stiff > 0:
                        df.at[idx, f'local_stiffness_zscore_w{window_size}'] = (grids['Stiffness'][y, x] - local_mean_stiff) / local_std_stiff

                # Spatial patterns - edge and corner detection
                # For x derivatives
                if 0 < x < width - 1:
                    # Interior point - use central difference
                    Ixx = grids['Stiffness'][y, x + 1] + grids['Stiffness'][y, x - 1] - 2 * grids['Stiffness'][y, x]
                elif x == 0:
                    # Left boundary - use forward difference
                    Ixx = 2 * grids['Stiffness'][y, x + 1] - 2 * grids['Stiffness'][y, x]
                else:  # x == width - 1
                    # Right boundary - use backward difference
                    Ixx = 2 * grids['Stiffness'][y, x - 1] - 2 * grids['Stiffness'][y, x]

                # For y derivatives
                if 0 < y < height - 1:
                    # Interior point - use central difference
                    Iyy = grids['Stiffness'][y + 1, x] + grids['Stiffness'][y - 1, x] - 2 * grids['Stiffness'][y, x]
                elif y == 0:
                    # Top boundary - use forward difference
                    Iyy = 2 * grids['Stiffness'][y + 1, x] - 2 * grids['Stiffness'][y, x]
                else:  # y == height - 1
                    # Bottom boundary - use backward difference
                    Iyy = 2 * grids['Stiffness'][y - 1, x] - 2 * grids['Stiffness'][y, x]

                # For mixed derivatives
                if 0 < x < width - 1 and 0 < y < height - 1:
                    # Interior point - use standard formula
                    Ixy = (grids['Stiffness'][y + 1, x + 1] - grids['Stiffness'][y + 1, x - 1] -
                           grids['Stiffness'][y - 1, x + 1] + grids['Stiffness'][y - 1, x - 1]) / 4
                else:
                    Ixy = 0  # Set to zero at boundary

                # Harris corner response
                k = 0.05  # Typical value
                det_M = Ixx * Iyy - Ixy ** 2
                trace_M = Ixx + Iyy
                harris_response = det_M - k * trace_M ** 2
                df.at[idx, 'stiffness_harris_response'] = harris_response

                # Edge response (from Sobel-like operator)
                edge_response = np.sqrt(Ixx ** 2 + Iyy ** 2)
                df.at[idx, 'stiffness_edge_response'] = edge_response

                # Local curvature features of surface
                # For x second derivatives
                if 0 < x < width - 1:
                    # Interior point - use central difference
                    Zxx = filled_position_grid[y, x + 1] + filled_position_grid[y, x - 1] - 2 * \
                          filled_position_grid[y, x]
                elif x == 0:
                    # Left boundary - use forward difference
                    Zxx = 2 * filled_position_grid[y, x + 1] - 2 * filled_position_grid[y, x]
                else:  # x == width - 1
                    # Right boundary - use backward difference
                    Zxx = 2 * filled_position_grid[y, x - 1] - 2 * filled_position_grid[y, x]

                # For y second derivatives
                if 0 < y < height - 1:
                    # Interior point - use central difference
                    Zyy = filled_position_grid[y + 1, x] + filled_position_grid[y - 1, x] - 2 * \
                          filled_position_grid[y, x]
                elif y == 0:
                    # Top boundary - use forward difference
                    Zyy = 2 * filled_position_grid[y + 1, x] - 2 * filled_position_grid[y, x]
                else:  # y == height - 1
                    # Bottom boundary - use backward difference
                    Zyy = 2 * filled_position_grid[y - 1, x] - 2 * filled_position_grid[y, x]

                # For mixed derivatives
                if 0 < x < width - 1 and 0 < y < height - 1:
                    # Interior point - use standard formula
                    Zxy = (filled_position_grid[y + 1, x + 1] - filled_position_grid[y + 1, x - 1] -
                           filled_position_grid[y - 1, x + 1] + filled_position_grid[y - 1, x - 1]) / 4
                else:
                    Zxy = 0

                # Mean and Gaussian curvature
                if 0 < x < width - 1:
                    Zx = (filled_position_grid[y, x + 1] - filled_position_grid[y, x - 1]) / 2
                elif x == 0:
                    Zx = filled_position_grid[y, x + 1] - filled_position_grid[y, x]
                else:  # x == width - 1
                    Zx = filled_position_grid[y, x] - filled_position_grid[y, x - 1]

                if 0 < y < height - 1:
                    Zy = (filled_position_grid[y + 1, x] - filled_position_grid[y - 1, x]) / 2
                elif y == 0:
                    Zy = filled_position_grid[y + 1, x] - filled_position_grid[y, x]
                else:  # y == height - 1
                    Zy = filled_position_grid[y, x] - filled_position_grid[y - 1, x]

                # Calculate terms for curvature
                E = 1 + Zx ** 2
                F = Zx * Zy
                G = 1 + Zy ** 2

                # Calculate Mean and Gaussian curvature
                mean_curvature = (E * Zyy - 2 * F * Zxy + G * Zxx) / (2 * (E * G - F ** 2) ** (3 / 2))
                gaussian_curvature = (Zxx * Zyy - Zxy ** 2) / (1 + Zx ** 2 + Zy ** 2) ** 2

                df.at[idx, 'surface_mean_curvature'] = mean_curvature
                df.at[idx, 'surface_gaussian_curvature'] = gaussian_curvature

                # Surface classification based on curvature signs
                if mean_curvature > 0 and gaussian_curvature > 0:
                    surface_type = 1  # Peak
                elif mean_curvature < 0 and gaussian_curvature > 0:
                    surface_type = 2  # Pit
                elif gaussian_curvature < 0:
                    surface_type = 3  # Saddle
                elif abs(mean_curvature) < 1e-6 and abs(gaussian_curvature) < 1e-6:
                    surface_type = 4  # Flat
                else:
                    surface_type = 5  # Ridge/valley

                df.at[idx, 'surface_type'] = surface_type

    return df


def interpolate_missing_values(grid):
    """
    Interpolate missing values in a grid using nearest neighbor interpolation.
    """
    # Check if the grid has any non-NaN values
    if np.all(np.isnan(grid)):
        return grid.copy()

    # Get coordinates of non-NaN values
    y_indices, x_indices = np.where(~np.isnan(grid))
    valid_points = np.column_stack((x_indices, y_indices))
    valid_values = grid[y_indices, x_indices]

    # Create grid coordinates for interpolation
    y_grid, x_grid = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    all_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Interpolate
    interpolated_values = griddata(valid_points, valid_values, all_points) # method='linear' default

    # Use nearest neighbor for remaining NaN values
    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        nearest_values = griddata(valid_points, valid_values, all_points[nan_mask], method='nearest')
        interpolated_values[nan_mask] = nearest_values

    # Reshape back to grid
    filled_grid = interpolated_values.reshape(grid.shape)

    return filled_grid

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
    offset = np.mean(df_input['forceZ'][df_input['isArrived_Festo'] == 1])
    scaling = np.std(df_input['forceZ'][df_input['isArrived_Festo'] == 1])
    #Z-score on force
    df_input['forceZ'] = (df_input['forceZ'] - offset)  / np.mean(df_input['forceZ'][df_input['isTouching_SMAC'] == 1][ -30:])  # / np.std(df_input['forceZ'][df_input['isArrived_Festo'] == 1].tolist())   #  # /

    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000
    #offset_p = np.mean(df_input['posz'][df_input['isArrived_Festo'] == 1])
    #scaling_p = np.std(df_input['posz'][df_input['isArrived_Festo'] == 1])
    #Z-score on position
    #df_input['posz'] = (df_input['posz'] - offset_p) / scaling_p

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

    (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke1,downstroke2, fi, pi, time_i, P_ss, offset,
     force_max, time_to_max, force_overshoot, peak_width, max_pos_rate,
     force_oscillation, force_relaxation, stiffness_ratio,
     jerk_max, zero_crossings_force, zero_crossings_position,
     force_rms, position_rms, damping_coefficient,
     spectral_centroid_force, spectral_entropy_force,
     spectral_centroid_position, spectral_entropy_position,
     impedance_ratio_lowfreq, coherence_LF, coherence_HF, elastic_coefficient) = tuple

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
        "Downstroke1": downstroke1,
        "Downstroke2": downstroke2,
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
        # New features
        "max_position": np.max(pi) if len(pi) > 0 else None,
        "hysteresis_area": np.trapezoid(fi, pi) if len(fi) > 0 and len(pi) > 0 else None,
        "energy_input": np.trapezoid(fi[:np.argmax(fi) + 1], pi[:np.argmax(fi) + 1]) if len(fi) > 0 and len(
            pi) > 0 else None,
        "jerk_max": jerk_max,
        "zero_crossings_force": zero_crossings_force,
        "zero_crossings_position": zero_crossings_position,
        "force_rms": force_rms,
        "position_rms": position_rms,
        "damping_coefficient": damping_coefficient,
        "spectral_centroid_force": spectral_centroid_force,
        "spectral_entropy_force": spectral_entropy_force,
        "spectral_centroid_position": spectral_centroid_position,
        "spectral_entropy_position": spectral_entropy_position,
        "impedance_ratio_lowfreq": impedance_ratio_lowfreq,
        "coherence_LF": coherence_LF,
        "coherence_HF": coherence_HF,
        "elastic_coeff": elastic_coefficient,
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


def create_df(path: str = 'DamasconeA/Dataset/*.csv', symmetric=False, symm_v=False, angle=0, offset = (49, 49)) -> DataFrame | None:
    # Load JSON file

    strings = path.split('/')
    with open(strings[0]+"/"+strings[1] + "/phantom_metadata.json", "r") as file:
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
    combined_df = pd.concat(non_empty_df_list, ignore_index=True)
    combined_df = compute_spatial_and_surface_features(combined_df)
    return combined_df
