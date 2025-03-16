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

    pz = (data['posz_s'][data['isArrived_Festo'] == 1]/1000).to_numpy() # From mm to m
    fz = (data['Fz_s'][data['isArrived_Festo'] == 1]).to_numpy()
    time = data['CPXEts'][data['isArrived_Festo'] == 1].to_numpy()

    fz_touch = (data['Fz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    pz_touch = (data['posz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    time_touch = data['CPXEts'][data['isTouching_SMAC'] == 1].to_numpy()

    time_i = np.arange(time[0], time[-1], 0.003)
    f= interp1d(time, fz, kind='cubic')

    fi = f(time_i)
    g = interp1d(time,pz, kind='cubic')
    pi = g(time_i)
    # Force Decay Rate (τ)
    decay_time = time_touch - time_touch[0]
    decay_force = np.log(fz_touch)  # Log transform to get a linear function like ln(F) = -t/tau + c
    tau, _, _, _, _ = linregress(decay_time, decay_force)
    tau = -1 / tau  # Convert to time constant

    # Steady-State Force (F_ss) - Mean force at end of signal
    F_ss = np.mean(fz_touch [-30:])

    # Hardness
    stiffness = data['forceZ'].max() / (data['posz'][data['forceZ'].idxmax()])

    # Slope of Initial Step Up & Final Step Down
    gradient = np.gradient(fz,time)
    upstroke = np.max(gradient)  # Max increase
    downstroke = np.min(gradient)  # Max decrease

    slope_diff = upstroke - downstroke
    slope_ratio = abs(upstroke / downstroke)

    # PSD
    freqs, psd = welch(fz, fs=1 / np.mean(np.diff(time)))
    power = np.sum(psd)
    psd_peak = freqs[np.argmax(psd)]

    # Entropy of the Force Signal
    prob_dist = np.histogram(data['forceZ'][data['isTouching_SMAC'] == 1], bins=30, density=True)[0]  # Probability distribution
    prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
    entropy = -np.sum(prob_dist * np.log2(prob_dist))  # Shannon entropy => measure of randomness (estimation of fluctuations in the signal)
    # Entropy of the position
    prob_dist = np.histogram(data['posz'][data['isTouching_SMAC'] == 1], bins=30, density=True)[0]  # Probability distribution
    prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
    entropy_p = -np.sum(prob_dist * np.log2(prob_dist))  # Shannon entropy => measure of randomness (estimation of fluctuations in the signal)

    return stiffness,tau,F_ss,power,entropy,entropy_p,psd_peak,upstroke,downstroke, slope_diff,slope_ratio,fi,pi,time_i


def find_inclusions(json_data):
    circles = json_data["Inclusions"]
    c = [(circle["Position"][0] + 50, circle["Position"][1] + 50) for circle in circles]
    r = [circle["Diameter"]/2 for circle in circles]
    return c, r

def get_label(posx, posy, centers, radii) -> int:
    for i in range(len(centers)):
        center_x, center_y = centers[i]
        radius = radii[i]
        # Check if point is within the circle
        if ((posx - center_x) ** 2 + (posy - center_y) ** 2) <= (radius-0.5) ** 2 :
            return (i // 5) + 1  # Assign label based on group of 5 circles
    return 0  # Default label


def organize_df(df_input: DataFrame, centers, radii) -> DataFrame | None:

    df_input['forceZ'] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    offset = np.mean(df_input['forceZ'][df_input['isArrived_Festo'] == 1].head(20))
    df_input['forceZ'] = (df_input['forceZ'] - offset) / np.std(df_input['forceZ'][df_input['isArrived_Festo'] == 1].tolist())
    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    posx = np.mean(df_input['posx'][df_input['isArrived_Festo'] == 1] + df_input['posx_d'][df_input['isArrived_Festo'] == 1] / 1000 )
    posy = np.mean(df_input['posy'][df_input['isArrived_Festo'] == 1] + df_input['posy_d'][df_input['isArrived_Festo'] == 1] / 1000 )

    if posx < 20 and posy < 20:
        return None


    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000

    stiffness, tau, F_ss, power, entropy, entropy_p, psd_peak, upstroke, downstroke, slope_diff, slope_ratio, fi,pi,time_i= extract_features(df_input) #change label

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
        "Entropy_p": entropy_p,
        "Upstroke": upstroke,
        "Downstroke": downstroke,
        "Slope diff": slope_diff,
        "Slope ratio": slope_ratio,
        "Dominant Frequency": psd_peak,
        #"Fz": [df_input['forceZ'].tolist()],
        #"t": [df_input['CPXEts'].tolist()],
        #"Fz_s": [df_input['Fz_s'].tolist()],
        #"posz_s": [df_input['posz_s'].tolist()],
        #"Touching": [df_input['isTouching_SMAC'].tolist()],
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

    df_list=list()

    for file in csv_files:
        df = pd.read_csv(file)
        flag = (df['isTouching_SMAC'] == 0).all()
        if flag: continue
        df_ta = organize_df(df,centers, radii)
        if df_ta is not None and len(df_ta) > 0:
            df_ta.insert(0, "Source", file.split("_")[-1].split(".")[0])
            df_list.append(df_ta)

    return pd.concat(df_list,ignore_index=True)