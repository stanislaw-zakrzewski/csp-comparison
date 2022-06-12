import numpy as np
import matplotlib.pyplot as plt


def visualize_multiple_configurations_band_accuracy(bands, bands_data, configuration_names, plot_shape):
    fig, axs = plt.subplots(plot_shape[0], plot_shape[1])
    for x_dim in range(plot_shape[0]):
        for y_dim in range(plot_shape[1]):
            band_data_index = x_dim * plot_shape[1] + y_dim
            band_name = '{}Hz - {}Hz'.format(bands[band_data_index][0], bands[band_data_index][1])

            fig.suptitle('Vertically stacked subplots')
            for index, band_data in enumerate(bands_data):
                axs[x_dim, y_dim].plot(band_data[band_data_index][0], np.mean(band_data[band_data_index][1], 0))
            axs[x_dim, y_dim].axvline(0, linestyle='--', color='k', label='Onset')
            axs[x_dim, y_dim].axhline(0.5, linestyle='-', color='k', label='Chance')
            axs[x_dim, y_dim].set_xlabel('time (s)')
            axs[x_dim, y_dim].set_ylabel('classification accuracy')
            axs[x_dim, y_dim].set_ylim(0.4, 1)
            axs[x_dim, y_dim].legend(configuration_names)
            axs[x_dim, y_dim].set_title('Classification {}'.format(band_name))

    plt.show()
