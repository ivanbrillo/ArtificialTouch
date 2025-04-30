import matplotlib.pyplot as plt
import seaborn as sns

feature_list_time2 = [
    'Stiffness', 'Tau', 'Upstroke', 'Downstroke', 'Entropy', 'P_ss',
    'offset', 'time_to_max', 'force_overshoot', 'force_relaxation',  'stiffness_ratio', 'peak_width' ]

feature_list_hysteresis2 = [
    'loading_unloading_area_ratio', 'cubic_coefficient',
    'quartic_coefficient', 'loading_kurtosis', 'loading_nonlinearity',
    'loading_skewness', 'force_ratio_75_25', 'hysteresis_area', 'peak_position',
    # Polynomial coefficients
    'poly3_coef0', 'poly3_coef1', 'poly3_coef2', 'poly3_coef3',
    'poly4_coef0', 'poly4_coef1', 'poly4_coef2', 'poly4_coef3', 'poly4_coef4',
    'poly5_coef0', 'poly5_coef1', 'poly5_coef2', 'poly5_coef3', 'poly5_coef4',
    # Segmentation features
    'segment2_slope', 'segment3_slope', 'segment2_force_std', 'segment3_force_std',
    'segment2_skew', 'segment3_skew', 'loading_energy'
]

feature_list2 = feature_list_time2 + feature_list_hysteresis2

feature_list_all = [
    "Stiffness", "Tau", "Force Steady State", "Power", "Entropy", "Upstroke", "Downstroke1","Downstroke2",
    "Dominant Frequency", "P_ss", "offset", "force_max", "time_to_max", "force_overshoot",
    "force_relaxation", "stiffness_ratio", "force_oscillation", "peak_width", "max_pos_rate",
    "max_position", "hysteresis_area", "energy_input", "jerk_max", "zero_crossings_force",
    "zero_crossings_position", "force_rms", "position_rms", "damping_coefficient",
    "spectral_centroid_force", "spectral_entropy_force", "spectral_centroid_position",
    "spectral_entropy_position", "impedance_ratio_lowfreq", "peak_position",
    "coherence_LF", "coherence_HF", "elastic_coeff",
    "loading_energy", "loading_nonlinearity", "loading_unloading_area_ratio",
    "force_ratio_75_25", "loading_skewness", "loading_kurtosis",
    "poly3_coef0", "poly3_coef1", "poly3_coef2", "poly3_coef3",
    "poly4_coef0", "poly4_coef1", "poly4_coef2", "poly4_coef3", "poly4_coef4",
    "poly5_coef0", "poly5_coef1", "poly5_coef2", "poly5_coef3", "poly5_coef4", "poly5_coef5",
    "cubic_coefficient", "quartic_coefficient",
    "segment2_slope", "segment2_force_std", "segment2_skew",
    "segment3_slope", "segment3_force_std", "segment3_skew",
    'local_mean_Stiffness', 'local_std_Stiffness', 'local_skewness_Stiffness', 'local_kurtosis_Stiffness',
    'local_range_Stiffness', 'local_mean_Upstroke', 'local_std_Upstroke', 'local_skewness_Upstroke', 'local_kurtosis_Upstroke',
    'local_range_Upstroke', 'local_mean_Downstroke1', 'local_std_Downstroke1', 'local_skewness_Downstroke1', 'local_kurtosis_Downstroke1',
    'local_range_Downstroke1', 'local_mean_Downstroke2', 'local_std_Downstroke2', 'local_skewness_Downstroke2', 'local_kurtosis_Downstroke2',
    'local_range_Downstroke2', 'local_mean_Tau', 'local_std_Tau', 'local_skewness_Tau', 'local_kurtosis_Tau', 'local_range_Tau',
    'local_mean_time_to_max', 'local_std_time_to_max', 'local_skewness_time_to_max', 'local_kurtosis_time_to_max', 'local_range_time_to_max',
    "local_gradient_norm_stiffness",
    "stiffness_deviation_from_global", "position_deviation_from_global",
    "stiffness_deviation_from_global_median", "position_deviation_from_global_median",
    "stiffness_z_score", "position_z_score",
    "lbp_code_P8_R1", "lbp_code_P16_R2",
    "lbp_bin1_P8_R1", "lbp_bin2_P8_R1", "lbp_bin3_P8_R1",
    "lbp_val1_P8_R1", "lbp_val2_P8_R1", "lbp_val3_P8_R1",
    "lbp_bin1_P16_R2", "lbp_bin2_P16_R2", "lbp_bin3_P16_R2",
    "lbp_val1_P16_R2", "lbp_val2_P16_R2", "lbp_val3_P16_R2",
    "lbp_mean_P8_R1", "lbp_std_P8_R1", "lbp_entropy_P8_R1",
    "lbp_mean_P16_R2", "lbp_std_P16_R2", "lbp_entropy_P16_R2",
    "local_Ra_w3", "local_Rq_w3", "local_skewness_w3", "local_kurtosis_w3",
    "local_peak_height_w3", "local_valley_depth_w3", "local_stiffness_w3", "local_stiffness_zscore_w3",
    "local_Ra_w5", "local_Rq_w5", "local_skewness_w5", "local_kurtosis_w5",
    "local_peak_height_w5", "local_valley_depth_w5", "local_stiffness_w5", "local_stiffness_zscore_w5",
    "local_Ra_w7", "local_Rq_w7", "local_skewness_w7", "local_kurtosis_w7",
    "local_peak_height_w7", "local_valley_depth_w7", "local_stiffness_w7", "local_stiffness_zscore_w7",
    "stiffness_harris_response", "stiffness_edge_response",
    "surface_mean_curvature", "surface_gaussian_curvature", "surface_type",
    "stiffness_to_relaxation", "oscillation_to_max_force"
]


def plot_class_distribution(labels, ax, title):
    sns.countplot(x=labels, ax=ax, palette='Set2', hue=labels)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.legend(title='Label', loc='upper right', frameon=False)
