import numpy as np
import matplotlib.pyplot as plt


def visualize_multiple_configurations_band_accuracy(bands, bands_data, configuration_names, plot_shape):
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
