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
