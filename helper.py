import matplotlib.pyplot as plt
import seaborn as sns

feature_list_time = [
    'Stiffness', 'Tau', 'Upstroke', 'Downstroke', 'Entropy', 'P_ss',
    'offset', 'time_to_max', 'force_overshoot', 'force_relaxation',  'stiffness_ratio', 'peak_width' ]

feature_list_hysteresis = [
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

feature_list = feature_list_time + feature_list_hysteresis


def plot_class_distribution(labels, ax, title):
    sns.countplot(x=labels, ax=ax, palette='Set2', hue=labels)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.legend(title='Label', loc='upper right', frameon=False)
