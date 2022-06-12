bare_minimum_channels = ['C3', 'C4']
minimum_channels = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4']

all_available_channels = ['FC5', 'FC3', 'FC2', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1', 'FC2', 'FC4', 'FC6', 'C2', 'C4',
                          'C6', 'CP2', 'CP4', 'CP6']

bands = [(6, 8), (8, 10), (20, 22), (22, 24)]

subject_name = 's14'

import mne
from experiments.band_classification import process_bands, visualize_csp_filters
from visualization.time_dependent_accuracy import visualize_multiple_configurations_band_accuracy
import matplotlib.pyplot as plt

# Hide all debug logs from mne
mne.set_log_level('ERROR')

# Increase plot size
plt.rcParams['figure.figsize'] = [16, 8]

experiment_names = ['2 channels', '10 channels', '18 channels', '64 channels']
experiment_channel_selections = [
    process_bands(subject_name, bands, bare_minimum_channels, True),
    process_bands(subject_name, bands, minimum_channels, True),
    process_bands(subject_name, bands, all_available_channels, True),
    process_bands(subject_name, bands, reset=True)
]