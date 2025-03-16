from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


# from lib.latent_space_helper import create_colors, get_colors_list, scatter_plot_latent


class PlotLearning(Callback):
    def __init__(self, autoencoder=None, show_latent=False):
        super().__init__()
        self.autoencoder = autoencoder
        self.show_latent = show_latent
        self.metrics = {}

        # if show_latent:
        #    database, _, self.data = load_database("databse.pkl", False)
        #    colors, _ = create_colors("exposure time", database, None)
        #    self.all_colors = get_colors_list(database, colors, "exposure time")

    def on_train_begin(self, logs=None):
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs=None):
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        metrics = [x for x in logs if 'val' not in x]
        new_plt = 1 if self.show_latent else 0
        f, axs = plt.subplots(1, len(metrics) + new_plt, figsize=(15, 5))

        if len(metrics) + new_plt == 1:
            axs = (axs,)
            plt.ylim(0, 1)
        elif len(metrics) + new_plt == 2:
            axs[0].set_ylim((0, 1))
        else:
            axs_limit = axs if not self.show_latent else axs[:-1]
            for ax in axs_limit:
                ax.set_ylim((0, 10))

        pred = None
        if self.show_latent:
            pred = self.autoencoder.encoder.predict(self.data)
            if isinstance(pred, tuple):
                pred = pred[2].T  # if VAE
            else:
                pred = pred.T

        clear_output(wait=True)
        print("A")

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        # if self.show_latent:
        #    scatter_plot_latent(axs[len(metrics)], pred, all_colors=self.all_colors)
        #    axs[len(metrics)].grid()
        # axs[len(metrics)].scatter(pred[0], pred[1], )

        plt.tight_layout()
        plt.show()

