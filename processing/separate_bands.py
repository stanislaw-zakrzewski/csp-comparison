import numpy as np
from mne import Epochs, pick_types
from mne.decoding import CSP, EMS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from config import eeg_data_path
from data_classes.subject import Subject


def process(subject_name, bands, selected_channels, reg=None):
    tmin, tmax = -1., 4.
    event_id = dict(left=0, right=1)
    subject = Subject(subject_name, eeg_data_path)

    raw_signals = []
    for i in range(len(bands)):
        raw_signals.append(subject.get_raw_copy())

    if len(selected_channels) > 0:
        for raw_signal in raw_signals:
            for channel in subject.electrode_names:
                if channel not in selected_channels:
                    raw_signal.drop_channels([channel])

    filtered_raw_signals = []
    epochs = []
    epochs_train = []
    epochs_data = []
    epochs_data_train = []

    # Apply band-pass filter
    for index, band in enumerate(bands):
        filtered_raw_signals.append(
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=.5, h_trans_bandwidth=.5, fir_design='firwin',
                                      skip_by_annotation='edge'))

    picks = pick_types(filtered_raw_signals[0].info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    for index, band in enumerate(bands):
        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs.append(
            Epochs(filtered_raw_signals[index], subject.events, event_id, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True))
        epochs_train.append(epochs[index].copy().crop(tmin=1., tmax=3.))

        # %%
        # Classification with linear discrimant analysis

        # Define a monte-carlo cross-validation generator (reduce variance):
        epochs_data.append(epochs[index].get_data())
        epochs_data_train.append(epochs_train[index].get_data())
    labels = epochs[0].events[:, -1]
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train[0])

    # Assemble a classifier
    # classifier = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=1,
    #                            max_iter=1000)  # Originally: LinearDiscriminantAnalysis()
    classifier = LinearDiscriminantAnalysis()
    csp_n_components = 10 if len(selected_channels) == 0 else min(len(selected_channels), 10)
    csp = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    # clf = Pipeline([('CSP', csp), ('Classifier', classifier)])
    # scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

    # Printing the results
    # class_balance = np.mean(labels == labels[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
    #                                                           class_balance))

    # plot CSP patterns estimated on full data for visualization
    # csp.fit_transform(epochs_data, labels)

    # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

    # %%
    # Look at performance over time

    sfreq = raw_signals[0].info['sfreq']
    w_length = int(sfreq)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data[0].shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = []
        x_test_csp = []
        for edt in epochs_data_train:
            if len(x_train_csp) > 0:
                x_train_csp = np.concatenate((x_train_csp, csp.fit_transform(edt[train_idx], y_train)), axis=1)
                x_test_csp = np.concatenate((x_test_csp, csp.transform(edt[test_idx])), axis=1)
            else:
                x_train_csp = csp.fit_transform(edt[train_idx], y_train)
                x_test_csp = csp.transform(edt[test_idx])

        # fit classifier
        classifier.fit(x_train_csp, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            x_test_csp = []
            for edt in epochs_data:
                if len(x_test_csp) > 0:
                    x_test_csp = np.concatenate(
                        (x_test_csp, csp.transform(edt[test_idx][:, :, n:(n + w_length)])),
                        axis=1)
                else:
                    x_test_csp = csp.transform(edt[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(classifier.score(x_test_csp, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin

    # plt.figure()
    # plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    # plt.axvline(0, linestyle='--', color='k', label='Onset')
    # plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    # plt.xlabel('time (s)')
    # plt.ylabel('classification accuracy')
    # plt.title('Classification score over time')
    # plt.legend(loc='lower right')
    # plt.show()

    return w_times, scores_windows, csp, epochs[0].info

# chosen_bands = [(10., 12.)]
#
# chosen_bands2 = [(20., 22.)]
# a1, a2 = process('s14', chosen_bands)
# b1, b2 = process('s14', chosen_bands2)
# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# ax1.plot(a1, np.mean(a2, 0))
# ax2.plot(b1, np.mean(b2, 0))
# plt.axvline(0, linestyle='--', color='k', label='Onset')
# plt.axhline(0.5, linestyle='-', color='k', label='Chance')
# plt.xlabel('time (s)')
# plt.ylabel('classification accuracy')
# plt.title('Classification score over time')
# plt.legend(loc='lower right')
# plt.show()
# plt.figure()
# plt.subplot(211,a1, np.mean(a2, 0), label='Score1')
# plt.subplot(212,b1, np.mean(b2, 0), label='Score2')
# plt.axvline(0, linestyle='--', color='k', label='Onset')
# plt.axhline(0.5, linestyle='-', color='k', label='Chance')
# plt.xlabel('time (s)')
# plt.ylabel('classification accuracy')
# plt.title('Classification score over time')
# plt.legend(loc='lower right')
# plt.show()


# values = {}
# v = []
# # for i in range(2):
# #     if i < 9:
# #         subject = 's0{}'.format(i + 1)
# #         values[subject] = np.mean(process(subject))
# #         v.append(np.mean(process(subject)))
# #     else:
# #         subject = 's{}'.format(i + 1)
# #         values[subject] = np.mean(process(subject))
# #         v.append(np.mean(process(subject)))
#
# for i in range(17):
#     f_min = i * 2 + 4
#     f_max = i * 2 + 6
#     name = '{}Hz-{}Hz'.format(f_min, f_max)
#     values[name] = np.mean(process('s14', f_min, f_max))
#     v.append(values[name])
#
# plt.bar(np.arange(1, 18), v)
# plt.show()
