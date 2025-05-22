from itertools import product

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def plot_class_distribution(labels, ax, title):
    sns.countplot(x=labels, ax=ax, palette='Set2', hue=labels)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.legend(title='Label', loc='upper right', frameon=False)


def prepare_xy(df, feature_list, label_column='label'):
    X = df[feature_list + ['posx', 'posy']]
    y = df[label_column]
    return X, y


def scale_dataset(df, feature_list):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_list])
    scaled_df = pd.DataFrame(scaled, columns=feature_list, index=df.index)

    scaled_df["posx"] = df["posx"]
    scaled_df["posy"] = df["posy"]

    return scaled_df


def get_hard_spots(X, y):
    hard_spots_index = y != 0
    hard_spots = X[hard_spots_index]
    y_hard = y[hard_spots_index]

    return hard_spots, y_hard


# Function to generate all combinations of binary and multiclass classifiers
def generate_combinations(df_features, models):
    combinations = []

    # Generate combinations for binary and multiclass model pairs
    for binary_model, multiclass_model in product(models, repeat=2):
        binary_model_params = binary_model['hyperparameters']
        multiclass_model_params = multiclass_model['hyperparameters']

        # Construct a dict for each combination
        param_dict = {
            'binary_classifier': [binary_model['model']],
            **{f'binary_classifier__{k}': v for k, v in binary_model_params.items()},
            'multiclass_classifier': [multiclass_model['model']],
            **{f'multiclass_classifier__{k}': v for k, v in multiclass_model_params.items()},
            'features': list(df_features['features'])
        }

        combinations.append(param_dict)

    return combinations
