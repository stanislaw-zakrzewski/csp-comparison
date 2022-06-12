import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import remove

from processing.separate_bands import process


def process_bands(subject_name, bands, selected_channels, reset=False):
    Path("experiments/temp_data").mkdir(parents=True, exist_ok=True)
    band_data = []
    if not reset:
        try:
            band_data = np.load('experiments/temp_data/band_data.npy', allow_pickle=True)
        except FileNotFoundError:
            print("No data found")

    # PROCESS BANDS
    if len(band_data) == 0:
        band_data = []
        for band in bands:
            band_name = '{} Hz - {} Hz'.format(band[0], band[1])
            print("Processing {} band".format(band_name))
            band_data.append(process(subject_name, [band], selected_channels))

    nd_band_data = np.asarray(band_data, dtype='object')
    np.save('experiments/temp_data/band_data', nd_band_data)
    return band_data


def visualize_csp_filters(bands, band_data, n_csp_components, plot_shape):
    fig1, axs1 = plt.subplots(plot_shape[0], plot_shape[1])
    for x_dim in range(plot_shape[0]):
        for y_dim in range(plot_shape[1]):
            band_data_index = x_dim * plot_shape[1] + y_dim
            band_name = '{}Hz - {}Hz'.format(bands[band_data_index][0], bands[band_data_index][1])

            element = band_data[band_data_index]
            mne_fig = element[2].plot_patterns(element[3], components=range(n_csp_components), ch_type='eeg',
                                               units='Patterns (AU)',
                                               size=1.5)
            plt.close(mne_fig)
            mne_fig.savefig('{}.png'.format(band_data_index))
            axs1[x_dim, y_dim].set_title('CSP {}'.format(band_name))
            axs1[x_dim, y_dim].imshow(plt.imread('{}.png'.format(band_data_index)))
            remove('{}.png'.format(band_data_index))
