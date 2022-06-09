import os

import numpy as np
from scipy.io import loadmat
from mne.io import concatenate_raws, RawArray
from mne import Epochs, pick_types, events_from_annotations, create_info
from mne.channels import make_standard_montage


class Subject:
    def __init__(self, subject_name, filepath):

        self.subject_mat = loadmat("{}/{}.mat".format(filepath, subject_name))
        self.subject_mat_sensor_locations = self.subject_mat['eeg'][0][0]['senloc']
        self.subject_mat_events = self.subject_mat['eeg'][0][0]['imagery_event']
        self.events = self.generate_events()

        data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(subject_name))

        self.electrode_names = list(make_standard_montage('biosemi64').get_positions()['ch_pos'].keys())

        mne_info = create_info(self.electrode_names, 512, 'eeg')

        raw1 = RawArray(data_left, mne_info)
        raw2 = RawArray(data_right, mne_info)

        montage = make_standard_montage('biosemi64')
        for i in range(0, 64):
            montage.dig[i + 3].update(r=self.subject_mat_sensor_locations[i])

        raw1.set_montage(montage)
        raw2.set_montage(montage)
        self.raw = concatenate_raws([raw1, raw2])

    def get_raw_copy(self):
        return self.raw.copy()

    def generate_events(self):
        event_localizations = np.where(self.subject_mat_events == 1)[1]
        data = []
        for i in event_localizations:
            data.append([i, 0, 0])
        for i in event_localizations:
            data.append([i + len(self.subject_mat_events[0]), 0, 1])
        return data
