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
    "Stiffness", "Tau", "Force Steady State", "Power", "Entropy", "Upstroke", "Downstroke1", "Downstroke2",
    "Dominant Frequency", "P_ss", "offset", "force_max", "time_to_max", "force_overshoot", "force_relaxation",
    "stiffness_ratio", "force_oscillation", "peak_width", "max_pos_rate", "max_position", "hysteresis_area",
    "energy_input", "jerk_max", "zero_crossings_force", "zero_crossings_position", "force_rms", "position_rms",
    "damping_coefficient", "spectral_centroid_force", "spectral_entropy_force", "spectral_centroid_position",
    "spectral_entropy_position", "impedance_ratio_lowfreq", "coherence_LF", "coherence_HF", "elastic_coeff",
    "contact_area", "adhesion_energy", "wavelet_energy_0", "wavelet_energy_1", "wavelet_energy_2",
    "wavelet_energy_3", "wavelet_energy_4", "stft_mean_freq",
    "imf_energy_0", "imf_energy_1", "imf_energy_2", "force_ptp", "position_ptp",
    "load_duration", "unload_duration", "load_unload_ratio", "loading_slope", "unloading_slope", "slope_symmetry",
    "curvature_peak", "slope_log_log", "activity", "mobility", "complexity_value", "hfd", "katz_fd", "hurst_exp",
    "tkeo_mean", "correlation_fp", "peak_ratio", "position_relaxation"
    # Curve features
    "peak_position", "hysteresis_area", "loading_energy", "loading_nonlinearity",
    "loading_unloading_area_ratio", "force_ratio_75_25", "loading_skewness", "loading_kurtosis",
    "poly3_coef0", "poly3_coef1", "poly3_coef2", "poly3_coef3",
    "poly4_coef0", "poly4_coef1", "poly4_coef2", "poly4_coef3", "poly4_coef4",
    "poly5_coef0", "poly5_coef1", "poly5_coef2", "poly5_coef3", "poly5_coef4",
    "segment2_slope", "segment3_slope", "segment2_force_std", "segment3_force_std",
    "segment2_skew", "segment3_skew", "cubic_coefficient", "quartic_coefficient",
    "strain_energy_density", "energy_dissipation_ratio",
    # Local statistics for each feature in features_list ['Stiffness', 'Upstroke', 'Downstroke1', 'Downstroke2', 'Tau', 'time_to_max']
    'local_mean_Stiffness', 'local_mean_Upstroke', 'local_mean_Downstroke1', 'local_mean_Downstroke2', 'local_mean_Tau', 'local_mean_time_to_max',
    'local_std_Stiffness', 'local_std_Upstroke', 'local_std_Downstroke1', 'local_std_Downstroke2', 'local_std_Tau', 'local_std_time_to_max',
    'local_skewness_Stiffness', 'local_skewness_Upstroke', 'local_skewness_Downstroke1', 'local_skewness_Downstroke2', 'local_skewness_Tau', 'local_skewness_time_to_max',
    'local_kurtosis_Stiffness', 'local_kurtosis_Upstroke', 'local_kurtosis_Downstroke1', 'local_kurtosis_Downstroke2', 'local_kurtosis_Tau', 'local_kurtosis_time_to_max',
    'local_range_Stiffness', 'local_range_Upstroke', 'local_range_Downstroke1', 'local_range_Downstroke2', 'local_range_Tau', 'local_range_time_to_max',
    'local_gradient_norm_Stiffness', 'local_gradient_norm_Upstroke', 'local_gradient_norm_Downstroke1', 'local_gradient_norm_Downstroke2', 'local_gradient_norm_Tau', 'local_gradient_norm_time_to_max',

    # Sobel gradient features
    'sobel_gradient_magnitude_Stiffness', 'sobel_gradient_magnitude_Upstroke', 'sobel_gradient_magnitude_Downstroke1', 'sobel_gradient_magnitude_Downstroke2', 'sobel_gradient_magnitude_Tau', 'sobel_gradient_magnitude_time_to_max',
    'sobel_gradient_direction_Stiffness', 'sobel_gradient_direction_Upstroke', 'sobel_gradient_direction_Downstroke1', 'sobel_gradient_direction_Downstroke2', 'sobel_gradient_direction_Tau', 'sobel_gradient_direction_time_to_max',

    # Laplacian features
    'laplacian_Stiffness', 'laplacian_Upstroke', 'laplacian_Downstroke1', 'laplacian_Downstroke2', 'laplacian_Tau', 'laplacian_time_to_max',

    # HOG features
    'hog_bin1', 'hog_bin2', 'hog_bin3',
    'hog_val1', 'hog_val2', 'hog_val3',
    'hog_mean', 'hog_std',

    # Global deviation features
    'stiffness_deviation_from_global', 'position_deviation_from_global',
    'stiffness_deviation_from_global_median', 'position_deviation_from_global_median',
    'stiffness_z_score', 'position_z_score',

    # LBP features for two parameter sets: P=8,R=1 and P=16,R=2
    'lbp_code_P8_R1', 'lbp_code_P16_R2',
    'lbp_bin1_P8_R1', 'lbp_bin2_P8_R1', 'lbp_bin3_P8_R1',
    'lbp_val1_P8_R1', 'lbp_val2_P8_R1', 'lbp_val3_P8_R1',
    'lbp_mean_P8_R1', 'lbp_std_P8_R1', 'lbp_entropy_P8_R1',
    'lbp_bin1_P16_R2', 'lbp_bin2_P16_R2', 'lbp_bin3_P16_R2',
    'lbp_val1_P16_R2', 'lbp_val2_P16_R2', 'lbp_val3_P16_R2',
    'lbp_mean_P16_R2', 'lbp_std_P16_R2', 'lbp_entropy_P16_R2',

    # Structure tensor features
    'structure_tensor_coherence', 'structure_tensor_orientation',
    'structure_tensor_lambda1', 'structure_tensor_lambda2',

    # Phase congruency features
    'phase_congruency_w3', 'phase_congruency_w5', 'phase_congruency_w7',

    # Distance weighted relationship features
    'distance_weighted_diff', 'boundary_likelihood',

    # Multi-scale edge response features
    'edge_response_scale1', 'edge_response_scale2', 'edge_response_scale3',

    # Local contrast normalization features (for each feature in features_list, multiple window sizes)
    'Stiffness_range_norm_w3', 'Stiffness_range_norm_w5', 'Stiffness_range_norm_w7', 'Stiffness_range_norm_w9',
    'Upstroke_range_norm_w3', 'Upstroke_range_norm_w5', 'Upstroke_range_norm_w7', 'Upstroke_range_norm_w9',
    'Downstroke1_range_norm_w3', 'Downstroke1_range_norm_w5', 'Downstroke1_range_norm_w7', 'Downstroke1_range_norm_w9',
    'Downstroke2_range_norm_w3', 'Downstroke2_range_norm_w5', 'Downstroke2_range_norm_w7', 'Downstroke2_range_norm_w9',
    'Tau_range_norm_w3', 'Tau_range_norm_w5', 'Tau_range_norm_w7', 'Tau_range_norm_w9',
    'time_to_max_range_norm_w3', 'time_to_max_range_norm_w5', 'time_to_max_range_norm_w7', 'time_to_max_range_norm_w9',

    # Center-surround difference features
    'center_surround_diff_c1_s3', 'center_surround_diff_c1_s5', 'center_surround_diff_c2_s3', 'center_surround_diff_c2_s5',
    'center_surround_ratio_c1_s3', 'center_surround_ratio_c1_s5', 'center_surround_ratio_c2_s3', 'center_surround_ratio_c2_s5',
    'center_surround_contrast_c1_s3', 'center_surround_contrast_c1_s5', 'center_surround_contrast_c2_s3', 'center_surround_contrast_c2_s5',

    # Local surface metrics
    'local_Ra_w3', 'local_Ra_w5', 'local_Ra_w7',
    'local_Rq_w3', 'local_Rq_w5', 'local_Rq_w7',
    'local_skewness_w3', 'local_skewness_w5', 'local_skewness_w7',
    'local_kurtosis_w3', 'local_kurtosis_w5', 'local_kurtosis_w7',
    'local_peak_height_w3', 'local_peak_height_w5', 'local_peak_height_w7',
    'local_valley_depth_w3', 'local_valley_depth_w5', 'local_valley_depth_w7',

    # Local stiffness anomaly features
    'local_stiffness_w3', 'local_stiffness_w5', 'local_stiffness_w7',
    'local_stiffness_zscore_w3', 'local_stiffness_zscore_w5', 'local_stiffness_zscore_w7',

    # Spatial pattern features
    'stiffness_harris_response', 'stiffness_edge_response',

    # Surface curvature features
    'surface_mean_curvature', 'surface_gaussian_curvature', 'surface_type',

    # Morphological features (for each threshold)
    'morph_area_mean_t0', 'morph_area_mean_t1', 'morph_area_mean_t2', 'morph_area_mean_t3',
    'morph_area_std_t0', 'morph_area_std_t1', 'morph_area_std_t2', 'morph_area_std_t3',
    'morph_perimeter_mean_t0', 'morph_perimeter_mean_t1', 'morph_perimeter_mean_t2', 'morph_perimeter_mean_t3',
    'morph_perimeter_std_t0', 'morph_perimeter_std_t1', 'morph_perimeter_std_t2', 'morph_perimeter_std_t3',
    'morph_euler_mean_t0', 'morph_euler_mean_t1', 'morph_euler_mean_t2', 'morph_euler_mean_t3',
    'morph_active_ratio_t0', 'morph_active_ratio_t1', 'morph_active_ratio_t2', 'morph_active_ratio_t3',
    'morph_border_touch_ratio_t0', 'morph_border_touch_ratio_t1', 'morph_border_touch_ratio_t2', 'morph_border_touch_ratio_t3',

    # Position context features
    'dist_to_grid_edge', 'is_grid_corner',

    # Entropy features
    'perm_entropy', 'sample_entropy', 'approx_entropy',

    # Fractal features
    'hurst_exponent', 'dfa_alpha', 'correlation_dim'
]


def plot_class_distribution(labels, ax, title):
    sns.countplot(x=labels, ax=ax, palette='Set2', hue=labels)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.legend(title='Label', loc='upper right', frameon=False)
