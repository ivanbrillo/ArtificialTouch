import pandas as pd
import glob
import numpy as np
import json
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.signal import freqz, coherence
from statsmodels.tsa.ar_model import AutoReg
from tqdm.notebook import tqdm
from scipy.fft import fft, fftshift,fftfreq
from scipy.integrate import trapezoid


def compute_transfer_function(fz, pz, time, max_order = 20):

    # Compute sampling frequency
    fs = 1 / (np.mean(np.diff(time)))

    # Compute FFT for force and position
    fft_fz = fft(fz)
    fft_pz = fft(pz)

    # Compute frequency axis (shifted to center)
    freqs = fftshift(fftfreq(len(fz), 1 / fs))

    # Compute Transfer Function H(f)
    # H(f) = FFT(Position) / FFT(Force)
    # Avoid division by zero
    H_f = np.zeros_like(fft_fz, dtype=complex)
    non_zero_force = np.abs(fft_fz) > np.max(np.abs(fft_fz)) * 1e-10
    H_f[non_zero_force] = fft_pz[non_zero_force] / fft_fz[non_zero_force]

    # Magnitude and Phase of Transfer Function
    H_mag = np.abs(H_f)
    H_phase = np.angle(H_f)

    # Find peaks in magnitude response
    peak_index = np.argmax(H_mag)
    peak_freq = freqs[peak_index]
    peak_mag = H_mag[peak_index]
    peak_phas = H_phase[peak_index]

    # Bandwidth (frequency range where magnitude is above half-power point)
    half_power_mag = np.max(H_mag) / np.sqrt(2)
    bandwidth_mask = H_mag >= half_power_mag
    bandwidth = np.abs(freqs[bandwidth_mask][-1] - freqs[bandwidth_mask][0])

    return {
        'transfer_function': H_f,
        'frequency_axis': freqs,
        'peak_frequency': peak_freq,
        'peak_magnitude': peak_mag,
        'peak_phase': peak_phas,
        'bandwidth': bandwidth,
    }

def extract_features(data):
    # Gaussian Smoothing
    data['Fz_s'] = gaussian_filter1d(data['forceZ'], sigma=2)
    data['posz_s'] = gaussian_filter1d(data['posz'], sigma=2)

    pz = (data['posz_s'][data['isArrived_Festo'] == 1] / 1000).to_numpy()  # From mm to m
    fz = (data['Fz_s'][data['isArrived_Festo'] == 1]).to_numpy()
    time = data['CPXEts'][data['isArrived_Festo'] == 1].to_numpy()

    fz_touch = (data['Fz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    pz_touch = (data['posz_s'][data['isTouching_SMAC'] == 1] / 1000).to_numpy()  # Convert to m
    time_touch = data['CPXEts'][data['isTouching_SMAC'] == 1].to_numpy()

    # Force Decay Rate (τ)
    decay_time = time_touch - time_touch[0]
    decay_force = np.log(fz_touch)  # Log transform to get a linear function like ln(F) = -t/tau + c
    tau, _, _, _, _ = linregress(decay_time, decay_force)
    tau = -1 / tau  # Convert to time constant

    # Steady-State Force (F_ss) and Position (P_ss)
    F_ss = np.mean(fz_touch[-30:])
    P_ss = np.mean(pz_touch[-30:])

    # Stiffness
    force_threshold = 0.1 * data['Fz_s'].max()
    force_onset_indices = np.where(data['Fz_s'] > force_threshold)[0]

    force_onset_idx = force_onset_indices[0]
    force_max_idx = data['Fz_s'].idxmax()

    # Position at force onset and max force
    pz_at_onset = data['posz_s'][force_onset_idx] / 1000  # Convert to meters
    pz_at_max = data['posz_s'][force_max_idx] / 1000  # Convert to meters

    # Calculate stiffness
    stiffness = data['Fz_s'].max() / (pz_at_max - pz_at_onset)

    # AR model
    # Detrend the data to focus on the dynamic response
    fz_ar = (data['forceZ'][data['isArrived_Festo'] == 1]).to_numpy()
    fz_detrended = fz_ar - np.mean(fz_ar)

    # Find optimal AR model order using AIC criterion (AKAIKE INFORMATION CRITERION)
    max_order = 30  # Limit maximum order
    aic_values = []

    for order in range(2, max_order + 1):
        # Use statsmodels AutoReg for AR model fitting
        model = AutoReg(fz_detrended, lags=order) # Lags: it's an AR process so the lags set the order of the model
        results = model.fit()
        aic_values.append(results.aic)

    # Find optimal order (minimum AIC)
    optimal_order = np.argmin(aic_values) + 2  # +2 because we started from order 2 (argmin returns an index that starts from 0)

    # Get AR coefficients using AutoReg with optimal order
    model = AutoReg(fz_detrended, lags=optimal_order)
    results = model.fit()

    # Extract AR coefficients
    ar_coeffs = results.params

    # Calculate frequency response
    freqs, h = freqz([1.0], np.concatenate(([1.0], ar_coeffs)), worN=1000) # worN specifies the number of frequency points at which to evaluate the frequency response

    # Convert digital frequency to analog frequency
    fs = 1 / (np.mean(np.diff(time)))  # Sampling frequency
    analog_freqs = freqs * fs / (2 * np.pi)

    # Find peaks in the response
    peak_index = np.argmax(np.abs(h))  # Index of the peak frequency
    peak_frequency = analog_freqs[peak_index]  # Peak frequency

    # Extract the poles of the AR model (roots of the denominator polynomial)
    poles = np.roots(np.concatenate(([1.0], ar_coeffs)))

    # Sort poles by magnitude (absolute value)
    sorted_poles = sorted(poles, key=abs, reverse=True)

    # Dominant pole (highest magnitude)
    dominant_pole = sorted_poles[0]
    dominant_pole_magnitude = abs(dominant_pole)
    dominant_pole_phase = np.angle(dominant_pole)

    # Fastest pole (smallest magnitude)
    fastest_pole = sorted_poles[-1]
    fastest_pole_magnitude = abs(fastest_pole)
    fastest_pole_phase = np.angle(fastest_pole)

    # Store model parameters
    ar_model = {
        'ar_coefficients': ar_coeffs.tolist(),
        'optimal_order': optimal_order,
        'poles': poles.tolist(),
        'peak_frequency': peak_frequency,
        'dominant pole': dominant_pole,
        'fastest pole': fastest_pole,
        'dominant pole magnitude': dominant_pole_magnitude,
        'dominant pole phase': dominant_pole_phase,
        'fastest pole magnitude': fastest_pole_magnitude,
        'fastest pole phase': fastest_pole_phase
    }

    # Compute Transfer Function
    transfer_function_model = compute_transfer_function(fz, pz, time)

    # Other features
    idx1 = np.where(data['Fz_s'] > 0.5)[0][0] if len(np.where(data['Fz_s'] > 0.5)[0]) > 0 else 0
    idx2 = np.where(data['Fz_s'] > 0.1)[0][0] if len(np.where(data['Fz_s'] > 0.1)[0]) > 0 else 0

    if idx1 > 0 and idx2 > 0 and idx1 != idx2:
        upstroke = (data['Fz_s'][idx1] - data['Fz_s'][idx2]) / (data['CPXEts'][idx1] - data['CPXEts'][idx2])
    else:
        upstroke = np.nan

    idx_back = np.where(data['Fz_s'] > 0.5)[0]
    idx1 = idx_back[-1] if len(idx_back) > 0 else 0

    idx_back = np.where(data['Fz_s'] > 0.1)[0]
    idx2 = idx_back[-1] if len(idx_back) > 0 else 0

    if idx1 > 0 and idx2 > 0 and idx1 != idx2:
        downstroke = (data['Fz_s'][idx1] - data['Fz_s'][idx2]) / (data['CPXEts'][idx1] - data['CPXEts'][idx2])
    else:
        downstroke = np.nan

    offset = np.mean(fz_touch[:5]) - np.mean(fz_touch[-10:]) if len(fz_touch) > 10 else np.nan

    # Entropy of the Force Signal
    if len(fz_touch) > 30:
        prob_dist = np.histogram(data['Fz_s'][data['isTouching_SMAC'] == 1], bins=30, density=True)[0]
        prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
        entropy = -np.sum(prob_dist * np.log2(prob_dist)) if len(prob_dist) > 0 else np.nan
    else:
        entropy = np.nan

    # New Interaction Features
    cross_corr = np.correlate(fz, pz, mode='full')
    max_corr = np.max(cross_corr)
    phase_shift = np.argmax(cross_corr) - len(fz) // 2
    work_done = trapezoid(fz * np.gradient(pz,time), time)
    coherence_values, _ = coherence(fz, pz, fs)
    mean_coherence = np.mean(coherence_values)

    interaction_features = {
        'max_cross_correlation': max_corr,
        'phase_shift': phase_shift,
        'work_done': work_done,
        'mean_coherence': mean_coherence
    }

    return stiffness, tau, F_ss, entropy, upstroke, downstroke,fz, pz, time, P_ss, offset, pz_at_onset, pz_at_max, ar_model,h,analog_freqs, transfer_function_model, interaction_features


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
        if ((posx - center_x) ** 2 + (posy - center_y) ** 2) <= (radius - 0.7) ** 2:
            return (i // 5) + 1  # Assign label based on group of 5 circles
    return 0  # Default label


def organize_df(df_input: DataFrame, centers, radii) -> DataFrame | None:
    df_input['forceZ'] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    offset = np.mean(df_input['forceZ'][df_input['isArrived_Festo'] == 1].head(20))
    df_input['forceZ'] = (df_input['forceZ'] - offset) / np.mean(df_input['forceZ'][df_input['isTouching_SMAC'] == 1][-30:])
    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    posx = np.mean(df_input['posx'][df_input['isArrived_Festo'] == 1] + df_input['posx_d'][df_input['isArrived_Festo'] == 1] / 1000)
    posy = np.mean(df_input['posy'][df_input['isArrived_Festo'] == 1] + df_input['posy_d'][df_input['isArrived_Festo'] == 1] / 1000)

    if posx < 20 and posy < 20:
        return None

    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000

    stiffness, tau, F_ss, entropy, upstroke, downstroke, fz, pz, time, P_ss, offset, pz_at_onset, pz_at_max, ar_model, h,analog_freqs, transfer_function_model, interaction_features = extract_features(df_input)

    label = get_label(posx, posy, centers, radii)

    new_df = DataFrame({
        "posx": posx,
        "posy": posy,
        "posz": [df_input['posz'].tolist()],
        "Force RAW": [df_input['forceZ'][df_input['isArrived_Festo'] == 1].tolist()],
        "Stiffness": stiffness,
        "Tau": tau,
        "Force Steady State": F_ss,
        "Entropy": entropy,
        "Upstroke": upstroke,
        "Downstroke": downstroke,
        "P_ss": P_ss,
        "offset": offset,
        "Max Cross Correlation": interaction_features['max_cross_correlation'],
        "Phase shift": interaction_features['phase_shift'],
        "Work": interaction_features['work_done'],
        "Mean Coherence": interaction_features['mean_coherence'],
        "t": [df_input['CPXEts'].tolist()],
        #"Fz_s": [df_input['Fz_s'].tolist()],
        #"posz_s": [df_input['posz_s'].tolist()],
        "Hforce_AR": [h],
        "Analog Freqs AR": [analog_freqs],
        "Dominant Pole": ar_model['dominant pole'],
        "Fastest Pole": ar_model['fastest pole'],
        "Dominant Pole Magnitude": ar_model['dominant pole magnitude'],
        "Dominant Pole Phase": ar_model['dominant pole phase'],
        "Fastest Pole Magnitude": ar_model['fastest pole magnitude'],
        "Fastest Pole Phase": ar_model['fastest pole phase'],
        "H": [transfer_function_model['transfer_function']],
        "Freq_axis": [transfer_function_model['frequency_axis']],
        "Peak_F": transfer_function_model['peak_frequency'],
        "Peak_M": transfer_function_model['peak_magnitude'],
        "Peak_P": transfer_function_model['peak_phase'],
        "Bandwidth": transfer_function_model['bandwidth'],
        #"Fz": [fz],
        #"Pz": [pz],
        "Time": [time],
        "Position at Force Onset": pz_at_onset,
        "Position at Force Max": pz_at_max,
        "AR Order": ar_model['optimal_order'],
        "AR Coefficients": [ar_model['ar_coefficients']],
        "AR Poles": [ar_model['poles']],
        "AR Peak Frequency": [ar_model['peak_frequency']],
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
    pbar = tqdm(total=len(csv_files))
    pbar.set_description("Processing : ")
    pbar.update(1)
    for file in csv_files:
        df = pd.read_csv(file)
        flag = (df['isTouching_SMAC'] == 0).all()
        if flag: continue
        df_ta = organize_df(df, centers, radii)
        if df_ta is not None and len(df_ta) > 0:
            df_ta.insert(0, "Source", file.split("_")[-1].split(".")[0])
            df_list.append(df_ta)
        pbar.update(1)
    return pd.concat(df_list, ignore_index=True)


def visualize_ar_model(df_row):
    """
    Visualize the AR model fit and frequency response for a specific data row
    """
    # Extract data
    fz = df_row['Force RAW']
    t = df_row['Time']

    # Get AR model parameters
    ar_coeffs = df_row['AR Coefficients']
    optimal_order = df_row['AR Order']

    # Detrend
    fz_detrended = fz - np.mean(fz)

    # Simulate AR model response
    ar_sim = np.zeros_like(fz_detrended)

    # Use the actual signal for initialization
    ar_sim[:optimal_order] = fz_detrended[:optimal_order]
    for i in range(optimal_order, len(fz_detrended)):
        ar_sim[i] = sum(ar_coeffs[j] * fz_detrended[i - j - 1] for j in range(len(ar_coeffs)))

    # Add mean back
    ar_sim = ar_sim + np.mean(fz)

    # Calculate frequency response
    freqs, h = freqz([1.0], np.concatenate(([1.0], ar_coeffs)), worN=1000)
    fs = 1 / (np.mean(np.diff(t)))  # Sampling frequency
    analog_freqs = freqs * fs / (2 * np.pi)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Time domain plot
    axes[0].plot(t, fz, 'b-', label='Original Force')
    axes[0].plot(t, ar_sim, 'r--', label=f'AR({optimal_order}) Model')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Force')
    axes[0].set_title('Time Domain Comparison: Original Force vs AR Model')
    axes[0].grid(True)
    axes[0].legend()

    # Frequency domain plot
    axes[1].plot(analog_freqs, 20 * np.log10(np.abs(h)), 'g-')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Frequency Response of AR Model')
    axes[1].grid(True)

    # Mark resonant frequencies
    freq = df_row['AR Peak Frequency']
    axes[1].axvline(x=freq, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    return fig