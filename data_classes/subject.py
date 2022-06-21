import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, RawArray
from scipy.io import loadmat

from config import eeg_data_path
from preprocessing.preprocess_subject import preprocess_subject


class Subject:
    def __init__(self, subject_name, filepath, with_rest=False):

        self.subject_mat = loadmat("{}/{}.mat".format(filepath, subject_name))
        self.subject_mat_sensor_locations = self.subject_mat['eeg'][0][0]['senloc']
        self.subject_mat_events = self.subject_mat['eeg'][0][0]['imagery_event']

        self.electrode_names = list(make_standard_montage('biosemi64').get_positions()['ch_pos'].keys())
        self.bad_trials = self.subject_mat['eeg'][0][0]['bad_trial_indices']
        self.bad_trials_voltage_left = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][0][0][
            0].flatten()
        self.bad_trials_voltage_right = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][0][0][
            1].flatten()
        self.bad_trials_mi_left = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][1][0][
            0].flatten()
        self.bad_trials_mi_right = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][1][0][
            1].flatten()
        mne_info = create_info(self.electrode_names, 512, 'eeg')

        self.events = self.generate_events(with_rest)

        try:
            data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(subject_name))
        except FileNotFoundError:
            preprocess_subject(['{}/{}.mat'.format(eeg_data_path, subject_name)])
            data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(subject_name))

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

    def generate_events(self, with_rest=False):
        event_localizations = np.where(self.subject_mat_events == 1)[1]
        data = []
        for index, value in enumerate(event_localizations):
            if index not in self.bad_trials_mi_left and index not in self.bad_trials_voltage_left:
                if with_rest:
                    data.append([value - 1023, 0, 0])
                    data.append([value + 512, 0, 1])
                else:
                    data.append([value, 0, 0])
        for index, value in enumerate(event_localizations):
            if index not in self.bad_trials_mi_right and index not in self.bad_trials_voltage_right:
                if with_rest:
                    data.append([value + len(self.subject_mat_events[0]) - 1023, 0, 0])
                    data.append([value + len(self.subject_mat_events[0]) + 512, 0, 2])
                else:
                    data.append([value + len(self.subject_mat_events[0]), 0, 1])
        return data
