import matplotlib.pyplot as plt
import numpy as np

from processing.separate_bands import process

bands = [(6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28),
         (28, 30), (30, 32), (32, 34), (34, 36), (36, 38), (38, 40), (40, 42)]

band_data = []
for band in bands:
    band_name = '{}Hz - {}Hz'.format(band[0], band[1])
    print("Started processing {} band".format(band_name))
    band_data.append(process('s14', [band]))
    print("Finish processing {} band".format(band_name))

fig, axs = plt.subplots(3, 6)

for x_dim in range(len(axs)):
    for y_dim in range(len(axs[0])):
        band_data_index = x_dim * len(axs[0]) + y_dim
        band_name = '{}Hz - {}Hz'.format(bands[band_data_index][0], bands[band_data_index][1])

        fig.suptitle('Vertically stacked subplots')
        axs[x_dim, y_dim].plot(band_data[band_data_index][0], np.mean(band_data[band_data_index][1], 0))
        axs[x_dim, y_dim].axvline(0, linestyle='--', color='k', label='Onset')
        axs[x_dim, y_dim].axhline(0.5, linestyle='-', color='k', label='Chance')
        axs[x_dim, y_dim].set_xlabel('time (s)')
        axs[x_dim, y_dim].set_ylabel('classification accuracy')
        axs[x_dim, y_dim].set_ylim(0.4, 1)
        axs[x_dim, y_dim].set_title('Classification {}'.format(band_name))
plt.show()
