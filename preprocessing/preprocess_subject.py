import numpy as np
from pathlib import Path
from scipy.io import loadmat


def preprocess_subject(subject_paths):
    subject_count = len(subject_paths)
    Path("preprocessed_data/subjects").mkdir(parents=True, exist_ok=True)
    for index, subject_path in enumerate(subject_paths):
        print("Started processing {} of {} subjects".format(index + 1, subject_count))

        subject = loadmat(subject_path)
        data_left = subject['eeg'][0][0]['imagery_left'][0:64]
        data_right = subject['eeg'][0][0]['imagery_right'][0:64]

        averages_left = []
        averages_right = []
        for i in range(64):
            averages_left.append([])
            averages_right.append([])

        for i in range(data_left.shape[1]):
            # TODO fix this to make it faster
            # concatenate((averages_left, full(64, [average(data_left[:, i])])), axis=1)
            # concatenate((averages_right, full(64, [average(data_right[:, i])])), axis=1)
            average_left = np.average(data_left[:, i])
            average_rigt = np.average(data_right[:, i])
            for j in range(64):
                averages_left[j].append(average_left)
                averages_right[j].append(average_rigt)

        data_left -= averages_left
        data_right -= averages_right
        np.save('preprocessed_data/subjects/{}'.format(subject_path.split('.')[-2].split('/')[-1]),
                [data_left, data_right])
        print("Finished processing {} of {} subjects".format(index + 1, subject_count))
