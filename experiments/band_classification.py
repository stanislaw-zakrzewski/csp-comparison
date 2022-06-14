import matplotlib.pyplot as plt
from os import remove

from processing.separate_bands import process


def process_bands(subject_name, bands, selected_channels=None, reg=None):
    if selected_channels is None:
        selected_channels = []

    band_data = []
    for band in bands:
        band_name = '{} Hz - {} Hz'.format(band[0], band[1])
        print("Processing {} band".format(band_name))
        band_data.append(process(subject_name, [band], selected_channels, reg))

    return band_data


def visualize_csp_filters(bands, band_data, n_csp_components, plot_shape):
    fig1, axs1 = plt.subplots(plot_shape[0], plot_shape[1])
    for x_dim in range(plot_shape[0]):
        for y_dim in range(plot_shape[1]):
            band_data_index = x_dim * plot_shape[1] + y_dim
            band_name = '{}Hz - {}Hz'.format(bands[band_data_index][0], bands[band_data_index][1])

            element = band_data[band_data_index]
            plt.title('okoo')
            mne_fig = element[2].plot_patterns(element[3], components=range(n_csp_components), ch_type='eeg',
                                               units='Patterns (AU)',
                                               size=1.5)
            plt.close(mne_fig)
            mne_fig.savefig('{}.png'.format(band_data_index))
            axs1[x_dim, y_dim].set_title('CSP {}'.format(band_name))
            axs1[x_dim, y_dim].imshow(plt.imread('{}.png'.format(band_data_index)))
            remove('{}.png'.format(band_data_index))
