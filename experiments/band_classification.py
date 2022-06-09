import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import remove

from processing.separate_bands import process

Path("experiments/temp_data").mkdir(parents=True, exist_ok=True)

bare_minimum_channels = ['C3', 'C4']
minimum_channels = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4']

all_available_channels = ['FC5', 'FC3', 'FC2', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1', 'FC2', 'FC4', 'FC6', 'C2', 'C4',
                          'C6', 'CP2', 'CP4', 'CP6']

bands = [(6, 8), (8, 10), (20, 22), (22, 24)]

bands = [(6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28),
         (28, 30), (30, 32), (32, 34), (34, 36), (36, 38), (38, 40), (40, 42)]


def process_bands(bands, selected_channels=[], reset=False):
    band_data = []
    if not reset:
        try:
            band_data = np.load('experiments/temp_data/band_data.npy', allow_pickle=True)
        except:
            print("No data found")

    # PROCESS BANDS
    if len(band_data) == 0:
        band_data = []
        for band in bands:
            band_name = '{}Hz - {}Hz'.format(band[0], band[1])
            print("Started processing {} band".format(band_name))
            band_data.append(process('s14', [band], selected_channels))
            print("Finish processing {} band".format(band_name))

    nd_band_data = np.asarray(band_data, dtype='object')
    np.save('experiments/temp_data/band_data', nd_band_data)
    return band_data


def visualize_multiple_configurations_band_accuracy(bands_data, configuration_names, plot_shape):
    fig2, axs2 = plt.subplots(plot_shape[0], plot_shape[1])
    for x_dim in range(len(axs2)):
        for y_dim in range(len(axs2[0])):
            band_data_index = x_dim * len(axs2[0]) + y_dim
            band_name = '{}Hz - {}Hz'.format(bands[band_data_index][0], bands[band_data_index][1])

            fig2.suptitle('Vertically stacked subplots')
            for index, band_data in enumerate(bands_data):
                axs2[x_dim, y_dim].plot(band_data[band_data_index][0], np.mean(band_data[band_data_index][1], 0))
            axs2[x_dim, y_dim].axvline(0, linestyle='--', color='k', label='Onset')
            axs2[x_dim, y_dim].axhline(0.5, linestyle='-', color='k', label='Chance')
            axs2[x_dim, y_dim].set_xlabel('time (s)')
            axs2[x_dim, y_dim].set_ylabel('classification accuracy')
            axs2[x_dim, y_dim].set_ylim(0.4, 1)
            axs2[x_dim, y_dim].legend(configuration_names)
            axs2[x_dim, y_dim].set_title('Classification {}'.format(band_name))

    plt.show()


def visualize_csp_filters(band_data, n_csp_components, plot_shape):
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


def visualize_bands(bands, selected_channels=[], reset=False):
    band_data = []
    if not reset:
        try:
            band_data = np.load('experiments/temp_data/band_data.npy', allow_pickle=True)
        except:
            print("No data found")

    # PROCESS BANDS
    if len(band_data) == 0:
        band_data = []
        for band in bands:
            band_name = '{}Hz - {}Hz'.format(band[0], band[1])
            print("Started processing {} band".format(band_name))
            band_data.append(process('s14', [band], selected_channels))
            print("Finish processing {} band".format(band_name))

    nd_band_data = np.asarray(band_data, dtype='object')
    np.save('experiments/temp_data/band_data', nd_band_data)

    # VISUALIZE TOP 2 CSP FILTERS
    fig1, axs1 = plt.subplots(3, 6)
    for x_dim in range(3):
        for y_dim in range(6):
            band_data_index = x_dim * 6 + y_dim
            band_name = '{}Hz - {}Hz'.format(bands[band_data_index][0], bands[band_data_index][1])

            element = band_data[band_data_index]
            csp_n_components = 2 if len(selected_channels) == 0 else min(len(selected_channels), 2)
            mne_fig = element[2].plot_patterns(element[3], components=range(csp_n_components), ch_type='eeg',
                                               units='Patterns (AU)',
                                               size=1.5)
            plt.close(mne_fig)
            mne_fig.savefig('{}.png'.format(band_data_index))
            axs1[x_dim, y_dim].set_title('CSP {}'.format(band_name))
            axs1[x_dim, y_dim].imshow(plt.imread('{}.png'.format(band_data_index)))
            remove('{}.png'.format(band_data_index))

    # VISUALIZE ACCURACY
    fig2, axs2 = plt.subplots(3, 6)
    for x_dim in range(len(axs2)):
        for y_dim in range(len(axs2[0])):
            band_data_index = x_dim * len(axs2[0]) + y_dim
            band_name = '{}Hz - {}Hz'.format(bands[band_data_index][0], bands[band_data_index][1])

            fig2.suptitle('Vertically stacked subplots')
            axs2[x_dim, y_dim].plot(band_data[band_data_index][0], np.mean(band_data[band_data_index][1], 0))
            axs2[x_dim, y_dim].axvline(0, linestyle='--', color='k', label='Onset')
            axs2[x_dim, y_dim].axhline(0.5, linestyle='-', color='k', label='Chance')
            axs2[x_dim, y_dim].set_xlabel('time (s)')
            axs2[x_dim, y_dim].set_ylabel('classification accuracy')
            axs2[x_dim, y_dim].set_ylim(0.4, 1)
            axs2[x_dim, y_dim].set_title('Classification {}'.format(band_name))

    plt.show()



# experiment_names = ['2 channels', '10 channels', '18 channels', '64 channels']
# experiment_channel_selections = [
#     process_bands(bands, bare_minimum_channels, True),
#     process_bands(bands, minimum_channels, True),
#     process_bands(bands, all_available_channels, True),
#     process_bands(bands, reset=True)
# ]
# visualize_multiple_configurations_band_accuracy(experiment_channel_selections, experiment_names, (6, 3))
# for index, experiment_name in enumerate(experiment_names):
#     visualize_csp_filters(experiment_channel_selections[index], 2, (6,3))
#
# # visualize_bands(bands, selected_channels=bare_minimum_channels, reset=True)
