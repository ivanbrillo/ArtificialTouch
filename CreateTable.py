from statistics import correlation
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
from scipy.optimize import curve_fit
from pywt import wavedec
from scipy.signal import stft
from PyEMD import EMD
from neurokit2 import complexity
from scipy import ndimage
from skimage import measure, morphology
import antropy
import nolds
from scipy.ndimage import sobel, laplace
from skimage.feature import hog

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

    # Calculate key indices once - reuse them throughout
    force_max = np.max(fi)
    force_max_idx = np.argmax(fi)
    steady_state_start_idx = min(force_max_idx + int(len(fi) * 0.05), len(fi) - 101)
    steady_state_force = np.mean(
        fi[steady_state_start_idx:steady_state_start_idx + 100]) if steady_state_start_idx < len(fi) - 100 else np.mean(
        fi[-100:])

    # Calculate differences and derivatives once (reuse for multiple features)
    dt = np.mean(np.diff(time_i))
    velocity = np.gradient(pi, time_i)
    acceleration = np.gradient(velocity, time_i)
    jerk = np.gradient(acceleration, time_i)

    # Calculate force thresholds once
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
    downstroke1 = None
    downstroke2 = None

    # Find indices efficiently
    low_idx_up = np.where(fi >= low_threshold)[0]
    mid_idx_up = np.where(fi >= mid_threshold)[0]
    high_idx_up = np.where(fi >= high_threshold)[0]

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
            if pi[force_max_idx] != pi[down1_idx]:
                downstroke1 = (force_max - fi[down1_idx]) / (pi[force_max_idx] - pi[down1_idx])

            if len(down_end_indices) > 0:
                down_end_idx = down_end_indices[0]
                if pi[down1_idx] != pi[down_end_idx]:
                    downstroke2 = (fi[down1_idx] - fi[down_end_idx]) / (pi[down1_idx] - pi[down_end_idx])

    # Calculate spectral features
    # Use smaller segment size for faster calculation
    freqs, psd = welch(ft, fs=1 / dt, nperseg=min(256, len(ft) // 4))
    power = np.mean(np.square(ft)) / (time_i[-1] - time_i[0])
    psd_peak = freqs[np.argmax(psd)] if len(psd) > 0 else None

    # Entropy
    # Use fewer bins for faster computation
    hist, _ = np.histogram(ft, bins=20, density=True)
    hist_positive = hist[hist > 0]
    entropy = -np.sum(hist_positive * np.log2(hist_positive)) if len(hist_positive) > 0 else None

    # Additional features
    time_to_max = time_i[force_max_idx] - time_i[0]
    force_overshoot = (force_max - F_ss) / F_ss if F_ss > 0 else 0

    # Peak width (simplified)
    above_th = np.where(fi > force_max * 0.9)[0]  # Use 90% of peak instead of exact peak
    peak_width = time_i[above_th[-1]] - time_i[above_th[0]] if len(above_th) > 0 else None

    # Position response rate
    max_pos_rate = np.max(np.abs(velocity[:force_max_idx + 1])) if force_max_idx > 0 else 0

    # Force oscillation
    force_oscillation = np.std(fi[steady_state_start_idx:steady_state_start_idx + 100]) if steady_state_start_idx < len(
        fi) - 100 else np.std(fi[-100:])

    # Force relaxation
    if len(ft) > 100:
        force_relaxation = (force_max - np.mean(ft[-50:])) / force_max if force_max > 0 else 0
    else:
        force_relaxation = 0

    # Position relaxation
    if len(pt) > 100:
        early_touch_position = pi[force_max_idx]
        late_touch_position = np.mean(pt[-50:])
        position_relaxation = (early_touch_position - late_touch_position) / early_touch_position if early_touch_position != 0 else 0
    else:
        position_relaxation = 0

    # Stiffness ratio (calculate once)
    stiffness_ratio = None
    if len(pi) > 100 and len(fi) > 100:
        # Early stiffness
        early_stiffness = None
        if np.any(fi >= 0.5 * force_max):
            early_idx = np.where(fi >= 0.5 * force_max)[0][0]
            start_idx = np.where(fi >= 0.1 * force_max)[0][0]
            if pi[early_idx] != pi[start_idx]:
                early_stiffness = (fi[early_idx] - fi[start_idx]) / (pi[early_idx] - pi[start_idx])

        # Late stiffness
        late_stiffness = None
        if np.any(fi >= 0.8 * force_max):
            late_idx = np.where(fi >= 0.8 * force_max)[0][0]
            if pi[force_max_idx] != pi[late_idx]:
                late_stiffness = (force_max - fi[late_idx]) / (pi[force_max_idx] - pi[late_idx])

        if early_stiffness and late_stiffness and early_stiffness != 0:
            stiffness_ratio = late_stiffness / early_stiffness

    # Max jerk (already computed)
    jerk_max = np.max(np.abs(jerk))

    # Force/position RMS
    force_rms = np.sqrt(np.mean(np.square(fi)))
    position_rms = np.sqrt(np.mean(np.square(pi)))

    # Zero-crossings (simplified)
    zero_crossings_force = np.sum(np.diff(np.signbit(fi - np.mean(fi))))
    zero_crossings_position = np.sum(np.diff(np.signbit(pi - np.mean(pi))))

    # Damping coefficient (optimized to avoid singular matrix errors)
    try:
        A = np.vstack([pi, velocity]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, fi, rcond=1e-10)
        elastic_coefficient, damping_coefficient = coeffs
    except:
        elastic_coefficient, damping_coefficient = None, None

    # Spectral features (FFT) - Use a power of 2 for faster FFT
    n = 2 ** int(np.log2(len(fi)))
    fi_fft = fft(fi[:n])
    pi_fft = fft(pi[:n])
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

    # Impedance ratio (low frequency)
    impedance_ratio_lowfreq = None
    low_freq_idx = (freq < 10) & (freq > 0)
    if np.any(low_freq_idx) and np.any(pi_mag[low_freq_idx] > 0):
        impedance_ratio_lowfreq = np.mean(fi_mag[low_freq_idx] / np.maximum(pi_mag[low_freq_idx], 1e-10))

    # Spectral coherence (optimized)
    coherence_LF = None
    coherence_HF = None
    if len(fi) > 128:
        f, Cxy = coherence(fi, pi, fs=1 / dt, nperseg=min(128, len(fi) // 4))
        low_freq_idx = f < 10
        high_freq_idx = f > 50
        if np.any(low_freq_idx):
            coherence_LF = np.mean(Cxy[low_freq_idx])
        if np.any(high_freq_idx):
            coherence_HF = np.mean(Cxy[high_freq_idx])

    # Wavelet features
    wavelet_relative_energies = [None] * 5
    try:
        from pywt import wavedec
        coeffs = wavedec(fi, 'db4', level=5)
        energies = [np.sum(np.square(c)) for c in coeffs]
        total_energy = sum(energies)
        if total_energy > 0:
            wavelet_relative_energies = [e / total_energy for e in energies]
            while len(wavelet_relative_energies) < 5:
                wavelet_relative_energies.append(0)
            wavelet_relative_energies = wavelet_relative_energies[:5]
    except:
        pass

    # STFT feature calculation
    stft_mean_freq = None
    try:
        from scipy.signal import stft
        f, t, Zxx = stft(fi, fs=1 / dt, nperseg=min(256, len(fi) // 4))
        magnitude = np.abs(Zxx)
        max_freq_indices = np.argmax(magnitude, axis=0)
        frequencies = f[max_freq_indices]
        stft_mean_freq = np.mean(frequencies)
    except:
        pass

    # EMD features
    imf_energy = [None, None, None]
    try:
        from PyEMD import EMD
        emd = EMD()
        imfs = emd(fi)
        imf_energy = []
        for i in range(min(3, len(imfs))):
            imf_energy.append(np.sum(np.square(imfs[i])))
        while len(imf_energy) < 3:
            imf_energy.append(0)
    except:
        pass


    # Contact area estimation
    contact_area = None
    if stiffness and stiffness > 0:
        R = 0.005  # Estimated radius in meters
        nu = 0.5  # Typical Poisson's ratio
        contact_area = np.pi * ((3 * (1 - nu ** 2) * force_max * R / (4 * stiffness)) ** (2 / 3))

    # Adhesion energy
    adhesion_energy = None
    try:
        retraction_force = fi[force_max_idx:]
        retraction_position = pi[force_max_idx:]
        adhesion_energy = -np.trapezoid(retraction_force, retraction_position)
    except:
        pass

    # Force peak-to-peak
    force_ptp = np.ptp(fi)

    # Position peak-to-peak
    position_ptp = np.ptp(pi)

    # Load duration calculation
    load_duration = time_i[force_max_idx] - time_i[0]

    # Unload duration calculation
    unload_duration = time_i[-1] - time_i[force_max_idx]

    # Load/unload ratio
    load_unload_ratio = load_duration / unload_duration if unload_duration > 0 else None

    # Loading slope
    loading_slope = (fi[force_max_idx] - fi[0]) / (pi[force_max_idx] - pi[0]) if pi[force_max_idx] != pi[0] else None

    # Unloading slope
    unloading_slope = (fi[-1] - fi[force_max_idx]) / (pi[-1] - pi[force_max_idx]) if pi[-1] != pi[
        force_max_idx] else None

    # Slope symmetry
    slope_symmetry = loading_slope / abs(unloading_slope) if unloading_slope and unloading_slope != 0 else None

    # Curvature peak
    curvature_peak = None
    try:
        d1 = np.gradient(fi, pi)
        d2 = np.gradient(d1, pi)
        curvature_peak = d2[force_max_idx]
    except:
        pass

    # Log-log slope calculation
    slope_log_log = None
    try:
        try:
            mask = pi > 0
            log_pi = np.log10(np.maximum(pi[mask], 1e-10))
            log_fi = np.log10(np.maximum(fi[mask], 1e-10))
            if len(log_pi) > 2:
                result = linregress(log_pi, log_fi)
                slope_log_log = result.slope
            else:
                pass
        except:
            pass
    except:
        pass

    # Hjorth parameters
    activity = np.var(fi)
    mobility = np.sqrt(np.var(np.diff(fi)) / activity) if activity > 0 else None
    complexity_value = np.sqrt(np.var(np.diff(np.diff(fi))) / np.var(np.diff(fi))) / mobility if mobility and np.var(
        np.diff(fi)) > 0 else None

    # Simplified Hurst exponent
    hurst_exp = None
    try:
        N = len(fi)
        Y = np.cumsum(fi - np.mean(fi))
        R = np.max(Y) - np.min(Y)
        S = np.std(fi)
        if S > 0:
            hurst_exp = np.log(R / S) / np.log(N)
    except:
        pass

    # Teager-Kaiser Energy Operator (simplified)
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
        # Compute permutation entropy with error handling
        try:
            # Permutation entropy with embedding dimension 3, delay 1
            perm_entropy = antropy.perm_entropy(fi, order=3, delay=1, normalize=True)
        except Exception:
            perm_entropy = np.nan
        try:
            # Sample_entropy
            sample_entropy = antropy.sample_entropy(fi)
        except Exception:
            sample_entropy = np.nan
    else:
        perm_entropy = np.nan
        sample_entropy = np.nan



    # Return all features in the original tuple format
    return (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke1, downstroke2, fi, pi, time_i, P_ss,
            force_max, time_to_max, force_overshoot, peak_width, max_pos_rate,
            force_oscillation, force_relaxation, stiffness_ratio,
            jerk_max, zero_crossings_force, zero_crossings_position,
            force_rms, position_rms, damping_coefficient,
            spectral_centroid_force, spectral_entropy_force,
            spectral_centroid_position, spectral_entropy_position,
            impedance_ratio_lowfreq, coherence_LF, coherence_HF, elastic_coefficient, contact_area, adhesion_energy,
            wavelet_relative_energies, stft_mean_freq, imf_energy, force_ptp, position_ptp, load_duration,
            unload_duration,
            load_unload_ratio, loading_slope, unloading_slope, slope_symmetry, curvature_peak,
            slope_log_log, activity, mobility, complexity_value,
            hurst_exp, tkeo_mean, correlation_fp, peak_ratio, position_relaxation, perm_entropy, sample_entropy)

def extract_curve_features(position, force):
    features = {}

    # --- 1. Peak detection ---
    peak_idx = np.argmax(force)
    features['peak_position'] = position[peak_idx]

    # --- 2. Hysteresis area ---
    # Ensure the curve is closed (append first point if necessary)
    hysteresis_area = np.nan
    if not np.array_equal(position[0], position[-1]) or not np.array_equal(force[0], force[-1]):
        position_closed = np.append(position, position[0])
        force_closed = np.append(force, force[0])
    else:
        position_closed = position
        force_closed = force
        hysteresis_area = 0.5 * abs(np.sum(
        position_closed[:-1] * force_closed[1:] - position_closed[1:] * force_closed[:-1]
    ))
        features['hysteresis_area'] = hysteresis_area

    # --- 3. Loading energy (area under loading curve) ---
    loading_energy = np.trapezoid(force[:peak_idx + 1], position[:peak_idx + 1])
    features['loading_energy'] = loading_energy

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
        loading_area = np.trapezoid(force[:peak_idx + 1], position[:peak_idx + 1])
        unloading_area = np.trapezoid(force[peak_idx:], position[peak_idx:])
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
        'segment2_skew', 'segment3_skew', 'cubic_coefficient', 'quartic_coefficient']

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

    # Create a list of features to analyze
    features_list = ['Stiffness', 'Upstroke', 'Downstroke1','Downstroke2','Tau','time_to_max']
    # Get the set of points that actually exist in the data
    existing_points = set(zip(df['posx'].astype(int), df['posy'].astype(int)))

    # Create grids for spatial analysis with interpolation for calculations
    grids = {}
    for feature in features_list:
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
            position_grid[y, x] = row['P_ss']

    # Interpolate missing values in position grid
    filled_position_grid = interpolate_missing_values(position_grid)

    # Compute all needed gradients and laplacians
    sobel_gx = {}
    sobel_gy = {}
    laplacians = {}
    for feature in features_list:
        sobel_gx[feature] = sobel(grids[feature], axis=1)
        sobel_gy[feature] = sobel(grids[feature], axis=0)
        laplacians[feature] = laplace(grids[feature])

    # Prepare depth map for LBP
    Stiffness_LBP_map = grids['Stiffness']

    # Define LBP parameters
    lbp_params = [
        (8, 1, 'uniform'),  # P=8, R=1
        (16, 2, 'uniform')  # P=16, R=2
    ]

    # Check if the stiffness map is valid and normalize it
    if not np.isnan(Stiffness_LBP_map).all():
        # Normalize Stiffness map to [0, 1] for LBP
        if np.ptp(Stiffness_LBP_map) > 0:
            norm_Stiffness_map = (Stiffness_LBP_map - np.min(Stiffness_LBP_map)) / np.ptp(Stiffness_LBP_map)
        else:
            norm_Stiffness_map = Stiffness_LBP_map - np.min(Stiffness_LBP_map)

        # Compute LBP maps
        lbp_maps = {}
        for P, R, method in lbp_params:
            try:
                lbp_maps[(P, R)] = local_binary_pattern(norm_Stiffness_map, P, R, method=method)
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

    # Morphological Descriptors from Multi-threshold Contact Maps
    # Extract timeseries data for each point
    force_timeseries = {}

    for idx, row in df.iterrows():
        x, y = int(row['posx']), int(row['posy'])
        if (x, y) in existing_points:
            force_values = np.array(row['Fi'])
            position_values = np.array(row['Pi'])
            if len(force_values) > 0 and len(position_values) > 0:
                force_timeseries[(x, y)] = force_values

    # Define thresholds as quantiles
    all_forces = np.concatenate([f for f in force_timeseries.values() if len(f) > 0])
    thresholds = np.quantile(all_forces, [0.25, 0.5, 0.75, 0.9])

    # Compute morphological descriptors for each point and threshold
    morphological_features = {point: {} for point in existing_points}

    # Maximum timesteps for sampling
    max_timestep= min(len(next(iter(force_timeseries.values()))), 4500) if force_timeseries else 0

    # Sample 10 time points evenly spaced throughout the range
    if max_timestep >= 10:
        sampled_timesteps = np.linspace(0, max_timestep - 1, 10, dtype=int)
    else:
        sampled_timesteps = range(max_timestep)  # Use all if less than 10 available

    # For each threshold
    for i, threshold in enumerate(thresholds):
        # Process only sampled time points
        for t in sampled_timesteps:
            # Create binary map with padding to avoid edge effects
            padded_binary_map = np.zeros((height + 2, width + 2), dtype=bool)

            for (x, y), forces in force_timeseries.items():
                if t < len(forces) and forces[t] > threshold:
                    grid_x, grid_y = x - minx + 1, y - miny + 1  # +1 for padding
                    if 0 <= grid_x - 1 < width and 0 <= grid_y - 1 < height:
                        padded_binary_map[grid_y, grid_x] = True

            # Only process if we have some active points
            if padded_binary_map.any():
                # Label connected components on padded map
                labeled_map, num_features = ndimage.label(padded_binary_map)

                # For each existing point, get local morphological descriptors
                for (x, y) in existing_points:
                    grid_x, grid_y = x - minx + 1, y - miny + 1  # +1 for padding

                    if 1 <= grid_x <= width and 1 <= grid_y <= height:
                        # Check if this point is active
                        is_active = padded_binary_map[grid_y, grid_x]

                        if is_active:
                            label = labeled_map[grid_y, grid_x]
                            component = (labeled_map == label)

                            # Check if component touches the border of the padded map
                            touches_border = (
                                    np.any(component[0, :]) or  # Top edge
                                    np.any(component[-1, :]) or  # Bottom edge
                                    np.any(component[:, 0]) or  # Left edge
                                    np.any(component[:, -1])  # Right edge
                            )

                            # Compute morphological descriptors
                            area = np.sum(component)
                            perimeter = measure.perimeter(component)
                            euler = measure.euler_number(component)

                            # Record features with border flag
                            key = f"morph_t{i}_time{i}"
                            morphological_features[(x, y)][key] = {
                                'area': area,
                                'perimeter': perimeter,
                                'euler': euler,
                                'active': 1,
                                'touches_border': int(touches_border)
                            }
                        else:
                            # Record that this point was not active
                            key = f"morph_t{i}_time{i}"
                            morphological_features[(x, y)][key] = {
                                'area': 0,
                                'perimeter': 0,
                                'euler': 0,
                                'active': 0,
                                'touches_border': 0
                            }

    # Add spatial position context for edge awareness
    for point in existing_points:
        x, y = point
        # Calculate distance to nearest edge
        dist_to_edge_x = min(x - minx, maxx - x)
        dist_to_edge_y = min(y - miny, maxy - y)
        dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)

        # Is this a corner point?
        is_corner = (dist_to_edge_x <= 1 and dist_to_edge_y <= 1)

        # Store these in our features
        morphological_features[point]['dist_to_edge'] = dist_to_edge
        morphological_features[point]['is_corner'] = int(is_corner)

    # Aggregate morphological features over time
    for point in existing_points:
        # For each threshold
        for i in range(len(thresholds)):
            # Collect values across time
            areas = []
            perimeters = []
            eulers = []
            actives = []
            border_touches = []
            key = f"morph_t{i}_time{i}"
            if key in morphological_features[point]:
                areas.append(morphological_features[point][key]['area'])
                perimeters.append(morphological_features[point][key]['perimeter'])
                eulers.append(morphological_features[point][key]['euler'])
                actives.append(morphological_features[point][key]['active'])
                border_touches.append(morphological_features[point][key]['touches_border'])

            # Compute statistics
            if areas:
                morphological_features[point][f'area_mean_t{i}'] = np.mean(areas)
                morphological_features[point][f'area_std_t{i}'] = np.std(areas)
                morphological_features[point][f'perimeter_mean_t{i}'] = np.mean(perimeters)
                morphological_features[point][f'perimeter_std_t{i}'] = np.std(perimeters)
                morphological_features[point][f'euler_mean_t{i}'] = np.mean(eulers)
                morphological_features[point][f'active_ratio_t{i}'] = np.mean(actives)
                morphological_features[point][f'border_touch_ratio_t{i}'] = np.mean(border_touches)
            else:
                # Default values if no data
                morphological_features[point][f'area_mean_t{i}'] = np.nan
                morphological_features[point][f'area_std_t{i}'] = np.nan
                morphological_features[point][f'perimeter_mean_t{i}'] = np.nan
                morphological_features[point][f'perimeter_std_t{i}'] = np.nan
                morphological_features[point][f'euler_mean_t{i}'] = np.nan
                morphological_features[point][f'active_ratio_t{i}'] = np.nan
                morphological_features[point][f'border_touch_ratio_t{i}'] = np.nan


    # Only process for EXISTING points
    for idx, row in df.iterrows():
        orig_x, orig_y = int(row['posx']), int(row['posy'])
        point = (orig_x, orig_y)

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
                for feature in features_list:
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
                        gx = (grids[feature][y, x + 1] - grids[feature][y, x - 1]) / 2
                    elif x == 0:
                        gx = grids[feature][y, x + 1] - grids[feature][y, x]
                    else:  # x == width - 1
                        gx = grids[feature][y, x] - grids[feature][y, x - 1]

                    if 0 < y < height - 1:
                        gy = (grids[feature][y + 1, x] - grids[feature][y - 1, x]) / 2
                    elif y == 0:
                        gy = grids[feature][y + 1, x] - grids[feature][y, x]
                    else:  # y == height - 1
                        gy = grids[feature][y, x] - grids[feature][y - 1, x]

                    df.at[idx, f'local_gradient_norm_{feature}'] = np.sqrt(gx ** 2 + gy ** 2)

                    # Compute features from Sobel gradients
                    gradient_magnitude = np.sqrt(sobel_gx[feature][y,x] ** 2 + sobel_gy[feature][y,x] ** 2)
                    gradient_direction = np.arctan2(sobel_gy[feature][y,x], sobel_gx[feature][y,x])

                    # Store the point-specific gradient information
                    df.at[idx, f'sobel_gradient_magnitude_{feature}'] = gradient_magnitude
                    df.at[idx, f'sobel_gradient_direction_{feature}'] = gradient_direction

                    # Assign Laplacian value
                    df.at[idx, f'laplacian_{feature}'] = laplacians[feature][y,x]


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

                # Compute structure tensors for directional information
                s_xx = sobel_gx[feature][y,x] * sobel_gx[feature][y,x]
                s_xy = sobel_gx[feature][y,x] * sobel_gy[feature][y,x]
                s_yy = sobel_gy[feature][y,x] * sobel_gy[feature][y,x]

                # Calculate eigenvalues of structure tensor
                tensor_trace = s_xx + s_yy
                tensor_det = s_xx * s_yy - s_xy * s_xy
                discriminant = np.sqrt(max(0, (tensor_trace ** 2) / 4 - tensor_det))
                lambda1 = tensor_trace / 2 + discriminant
                lambda2 = tensor_trace / 2 - discriminant

                # Compute coherence and orientation from structure tensor
                if (lambda1 + lambda2) > 0:
                    coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
                else:
                    coherence = 0

                orientation = 0.5 * np.arctan2(2 * s_xy, s_xx - s_yy) if s_xx != s_yy else np.pi / 4

                df.at[idx, 'structure_tensor_coherence'] = coherence
                df.at[idx, 'structure_tensor_orientation'] = orientation
                df.at[idx, 'structure_tensor_lambda1'] = lambda1
                df.at[idx, 'structure_tensor_lambda2'] = lambda2

                # Phase congruency approximation (simplified for computational efficiency)
                # Use multiple window sizes to capture different scales
                for window_size in [3, 5, 7]:
                    half_win = window_size // 2
                    y_min_pc = max(0, y - half_win)
                    y_max_pc = min(height - 1, y + half_win) + 1
                    x_min_pc = max(0, x - half_win)
                    x_max_pc = min(width - 1, x + half_win) + 1

                    patch = grids['Stiffness'][y_min_pc:y_max_pc, x_min_pc:x_max_pc]
                    if np.any(np.isnan(patch)):
                        patch = np.nan_to_num(patch, nan=np.nanmean(patch))

                    if patch.size > 1:
                        # Compute gradients in the patch
                        patch_gx = sobel(patch, axis=1)
                        patch_gy = sobel(patch, axis=0)
                        patch_gmag = np.sqrt(patch_gx ** 2 + patch_gy ** 2)

                        # Simple phase congruency metric - ratio of edge energy to total energy
                        total_energy = np.sum(np.abs(patch)) + 1e-10
                        edge_energy = np.sum(patch_gmag) + 1e-10
                        phase_congruency = edge_energy / total_energy

                        df.at[idx, f'phase_congruency_w{window_size}'] = phase_congruency

                # Distance-weighted relationship features
                # Define feature differences with neighboring points
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                feature_diffs = np.zeros(len(directions))
                dist_weights = np.zeros(len(directions))

                for i, (dy, dx) in enumerate(directions):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        center_val = grids['Stiffness'][y, x]
                        neighbor_val = grids['Stiffness'][ny, nx]
                        if not np.isnan(center_val) and not np.isnan(neighbor_val):
                            feature_diffs[i] = abs(center_val - neighbor_val)
                            # Inverse distance as weight
                            dist_weights[i] = 1.0 / np.sqrt(dx ** 2 + dy ** 2)

                # Normalize weights
                if np.sum(dist_weights) > 0:
                    dist_weights = dist_weights / np.sum(dist_weights)

                # Compute weighted differences
                weighted_diff = np.sum(feature_diffs * dist_weights)
                df.at[idx, 'distance_weighted_diff'] = weighted_diff

                # Boundary likelihood based on feature differentials
                # Higher values indicate higher likelihood of being at a boundary
                max_diff = np.max(feature_diffs)
                boundary_likelihood = max_diff / (np.mean(grids['Stiffness']) + 1e-10)
                df.at[idx, 'boundary_likelihood'] = boundary_likelihood

                # Multi-scale edge response features
                for scale in [1, 2, 3]:
                    # Define kernel size based on scale
                    kernel_size = 2 * scale + 1

                    # Skip if kernel would extend beyond grid
                    if (y - scale < 0 or y + scale >= height or
                            x - scale < 0 or x + scale >= width):
                        df.at[idx, f'edge_response_scale{scale}'] = np.nan
                        continue

                    # Extract and process sub-grid at this scale
                    sub_grid = grids['Stiffness'][y - scale:y + scale + 1, x - scale:x + scale + 1]

                    # Skip if sub_grid contains NaN values
                    if np.any(np.isnan(sub_grid)):
                        df.at[idx, f'edge_response_scale{scale}'] = np.nan
                        continue

                    # Compute Sobel response at this scale
                    sub_gx = sobel(sub_grid, axis=1)
                    sub_gy = sobel(sub_grid, axis=0)
                    edge_response = np.sqrt(sub_gx[scale, scale] ** 2 + sub_gy[scale, scale] ** 2)

                    df.at[idx, f'edge_response_scale{scale}'] = edge_response

                # Local Contrast Normalization
                # Multi-scale contrast features with different window sizes
                for window_size in [3, 5, 7, 9]:
                    half_win = window_size // 2

                    # Define window boundaries with bounds checking
                    y_min_win = max(0, y - half_win)
                    y_max_win = min(height - 1, y + half_win) + 1
                    x_min_win = max(0, x - half_win)
                    x_max_win = min(width - 1, x + half_win) + 1

                    # Process multiple features
                    for feature in features_list:
                        patch = grids[feature][y_min_win:y_max_win, x_min_win:x_max_win]
                        valid_patch = patch[~np.isnan(patch)]

                        if len(valid_patch) > 0:
                            # Local contrast (max-min range normalized)
                            local_range = np.max(valid_patch) - np.min(valid_patch)
                            if local_range > 0:
                                range_norm_value = (grids[feature][y, x] - np.min(valid_patch)) / local_range
                                df.at[idx, f'{feature}_range_norm_w{window_size}'] = range_norm_value
                            else:
                                df.at[idx, f'{feature}_range_norm_w{window_size}'] = 0

                # Center-surround differences at multiple scales
                for center_size in [1, 2]:
                    for surround_size in [3, 5]:
                        if surround_size <= center_size:
                            continue

                        # Define center region
                        y_min_center = max(0, y - center_size)
                        y_max_center = min(height - 1, y + center_size) + 1
                        x_min_center = max(0, x - center_size)
                        x_max_center = min(width - 1, x + center_size) + 1

                        # Define surround region
                        y_min_surr = max(0, y - surround_size)
                        y_max_surr = min(height - 1, y + surround_size) + 1
                        x_min_surr = max(0, x - surround_size)
                        x_max_surr = min(width - 1, x + surround_size) + 1

                        # Process for stiffness feature
                        center_patch = grids['Stiffness'][y_min_center:y_max_center, x_min_center:x_max_center]
                        valid_center = center_patch[~np.isnan(center_patch)]

                        # Create mask for surround (excluding center)
                        surround_patch = grids['Stiffness'][y_min_surr:y_max_surr, x_min_surr:x_max_surr]
                        center_mask = np.zeros_like(surround_patch, dtype=bool)
                        center_mask[
                        y_min_center - y_min_surr:y_max_center - y_min_surr,
                        x_min_center - x_min_surr:x_max_center - x_min_surr
                        ] = True

                        surround_values = surround_patch[~center_mask]
                        valid_surround = surround_values[~np.isnan(surround_values)]

                        if len(valid_center) > 0 and len(valid_surround) > 0:
                            center_mean = np.mean(valid_center)
                            surround_mean = np.mean(valid_surround)

                            # Center-surround difference
                            cs_diff = center_mean - surround_mean
                            df.at[idx, f'center_surround_diff_c{center_size}_s{surround_size}'] = cs_diff

                            # Center-surround ratio
                            if surround_mean != 0:
                                cs_ratio = center_mean / surround_mean
                                df.at[idx, f'center_surround_ratio_c{center_size}_s{surround_size}'] = cs_ratio
                            else:
                                df.at[idx, f'center_surround_ratio_c{center_size}_s{surround_size}'] = np.nan

                            # Center-surround contrast
                            cs_contrast = abs(center_mean - surround_mean) / (center_mean + surround_mean + 1e-10)
                            df.at[idx, f'center_surround_contrast_c{center_size}_s{surround_size}'] = cs_contrast

                # LOCAL SURFACE METRICS - multiple window sizes
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

                # Add morphological features
                for i in range(len(thresholds)):
                    df.at[idx, f'morph_area_mean_t{i}'] = morphological_features[point][f'area_mean_t{i}']
                    df.at[idx, f'morph_area_std_t{i}'] = morphological_features[point][f'area_std_t{i}']
                    df.at[idx, f'morph_perimeter_mean_t{i}'] = morphological_features[point][f'perimeter_mean_t{i}']
                    df.at[idx, f'morph_perimeter_std_t{i}'] = morphological_features[point][f'perimeter_std_t{i}']
                    df.at[idx, f'morph_euler_mean_t{i}'] = morphological_features[point][f'euler_mean_t{i}']
                    df.at[idx, f'morph_active_ratio_t{i}'] = morphological_features[point][f'active_ratio_t{i}']
                    df.at[idx, f'morph_border_touch_ratio_t{i}'] = morphological_features[point][f'border_touch_ratio_t{i}']

                # Add position context for edge awareness
                df.at[idx, 'dist_to_grid_edge'] = morphological_features[point]['dist_to_edge']
                df.at[idx, 'is_grid_corner'] = morphological_features[point]['is_corner']

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
    # Force normalization
    df_input['forceZ'] = (df_input['forceZ'] - offset)  / np.mean(df_input['forceZ'][df_input['isTouching_SMAC'] == 1][ -30:])  # / np.std(df_input['forceZ'][df_input['isArrived_Festo'] == 1].tolist())   #  # /

    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000
    offset_p = np.mean(df_input['posz'][df_input['isArrived_Festo'] == 1])
    # Position normalization
    df_input['posz'] = (df_input['posz'] - offset_p) /  np.mean(df_input['posz'][df_input['isTouching_SMAC'] == 1][ -30:])

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

    (stiffness, tau, F_ss, power, entropy, psd_peak, upstroke, downstroke1, downstroke2, fi, pi, time_i, P_ss,
    force_max, time_to_max, force_overshoot, peak_width, max_pos_rate,
    force_oscillation, force_relaxation, stiffness_ratio,
    jerk_max, zero_crossings_force, zero_crossings_position,
    force_rms, position_rms, damping_coefficient,
    spectral_centroid_force, spectral_entropy_force,
    spectral_centroid_position, spectral_entropy_position,
    impedance_ratio_lowfreq, coherence_LF, coherence_HF, elastic_coefficient, contact_area, adhesion_energy,
    wavelet_relative_energies, stft_mean_freq, imf_energy, force_ptp, position_ptp, load_duration,
    unload_duration,
    load_unload_ratio, loading_slope, unloading_slope, slope_symmetry, curvature_peak,
    slope_log_log, activity, mobility, complexity_value,
    hurst_exp, tkeo_mean, correlation_fp, peak_ratio, position_relaxation, perm_entropy, sample_entropy
     ) = tuple

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
        "contact_area": contact_area,
        "adhesion_energy": adhesion_energy,
        "wavelet_energy_0": wavelet_relative_energies[0],
        "wavelet_energy_1": wavelet_relative_energies[1],
        "wavelet_energy_2": wavelet_relative_energies[2],
        "wavelet_energy_3": wavelet_relative_energies[3],
        "wavelet_energy_4": wavelet_relative_energies[4],
        "stft_mean_freq": stft_mean_freq,
        "imf_energy_0": imf_energy[0],
        "imf_energy_1": imf_energy[1],
        "imf_energy_2": imf_energy[2],
        "force_ptp": force_ptp,
        "position_ptp": position_ptp,
        "load_duration": load_duration,
        "unload_duration": unload_duration,
        "load_unload_ratio": load_unload_ratio,
        "loading_slope": loading_slope,
        "unloading_slope": unloading_slope,
        "slope_symmetry": slope_symmetry,
        "curvature_peak": curvature_peak,
        "slope_log_log": slope_log_log,
        "activity": activity,
        "mobility": mobility,
        "complexity_value": complexity_value,
        "hurst_exp": hurst_exp,
        "tkeo_mean": tkeo_mean,
        "correlation_fp": correlation_fp,
        "peak_ratio": peak_ratio,
        "position_relaxation": position_relaxation,
        "perm_entropy": perm_entropy,
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
