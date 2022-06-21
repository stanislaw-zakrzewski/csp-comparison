import numpy as np
from mne import Epochs, pick_types, concatenate_epochs
from mne.decoding import CSP, EMS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from config import eeg_data_path
from data_classes.subject import Subject
import matplotlib.pyplot as plt

from visualization.csp_filters import visualize_csp_filters
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd
import scipy.stats as st

print(__doc__)


def create_confusion_matrix(n_classes, predictions_sets, corrects_sets):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for fold_index in range(len(predictions_sets)):
        predictions = predictions_sets[fold_index]
        corrects = corrects_sets[fold_index]
        for index in range(len(predictions)):
            confusion_matrix[predictions[index]][corrects[index]] += 1
    return confusion_matrix


def process(subject_name, bands, selected_channels, reg=None):
    tmin, tmax = 0., 2.
    event_id = dict(idle=0, left=1, right=2)
    subject = Subject(subject_name, eeg_data_path, True)

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
        epochs_train.append(epochs[index].copy().crop(tmin=0., tmax=2.))

        # %%
        # Classification with linear discrimant analysis

        # Define a monte-carlo cross-validation generator (reduce variance):
        epochs_data.append(epochs[index].get_data())
        epochs_data_train.append(epochs_train[index].get_data())
    labels = np.array(epochs[0].events[:, -1])
    # labels = np.where(labels == 2, 1, labels)
    o = np.max(labels)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train[0])

    # Assemble a classifier
    # classifier = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=1,
    #                            max_iter=1000)  # Originally: LinearDiscriminantAnalysis()
    classifier = LinearDiscriminantAnalysis()
    csp_n_components = 10 if len(selected_channels) == 0 else min(len(selected_channels), 10)
    csp = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)
    # csp.fit_transform(epochs_data[0], labels)
    # csp.plot_patterns(epochs[0].info, components=2, ch_type='eeg', units='Patterns (AU)', size=1.5)
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
    correct = {0: 0, 1: 0, 2: 0}
    all = {0: 0, 1: 0, 2: 0}
    all_predictions = []
    all_correct = []
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
        X_test_csp = csp.transform(epochs_data[0][test_idx])
        predictions = classifier.predict(X_test_csp)
        print(predictions, y_test)
        all_predictions.append(predictions)
        all_correct.append(y_test)

        for i in range(len(predictions)):
            pred = predictions[i]
            corr = y_test[i]
            if pred == corr: correct[corr] += 1
            all[corr] += 1

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
            # print(classifier.predict(x_test_csp))
            # oko += 1
            # print(y_test)
            score_this_window.append(classifier.score(x_test_csp, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin
    for i in range(3):
        print('Predictions:', i, correct[i], all[i])
    # plt.figure()
    # plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    # plt.axvline(0, linestyle='--', color='k', label='Onset')
    # plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    # plt.xlabel('time (s)')
    # plt.ylabel('classification accuracy')
    # plt.title('Classification score over time')
    # plt.legend(loc='lower right')
    # plt.show()

    return w_times, scores_windows, csp, epochs[0].info, all_predictions, all_correct


chosen_bands = [(10., 12.)]

chosen_bands2 = [(20., 22.)]
channels = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4']
confusion_matrices = []

# , gridspec_kw=dict(width_ratios=[4, 1, 0.2]))

all_scores = {'patient': [], 'accuracy': []}
chances = {'values': [], 'patients': []}
for patient_index in range(1, 3):
    print(patient_index)
    if patient_index < 10:
        patient_name = 's0{}'.format(patient_index)
    else:
        patient_name = 's{}'.format(patient_index)
    a1, a2, a3, a4, a5, a6 = process(patient_name, chosen_bands, channels)
    confusion_matrices.append(create_confusion_matrix(3, a5, a6))
    this_scores = []
    for i in range(len(a5)):
        all_scores['patient'].append(patient_index)
        oko = accuracy_score(a5[i], a6[i])
        print(oko)
        all_scores['accuracy'].append(accuracy_score(a5[i], a6[i]))
    chances['patients'].append(patient_index)
    chances['values'].append(st.binom.ppf(.9, len(a5[0]), .3) / len(a5[0]))

all_scores = pd.DataFrame(data=all_scores)
chances = pd.DataFrame(chances['values'], index=chances['patients']).transpose()

# fig, ax = plt.subplots()
# accuracy_plot = sns.barplot(x="patient", y="accuracy", data=all_scores, ci="sd",  ax=ax, zorder=0)
# sns.pointplot(data=chances, join=False, color='orange', ax=ax, zorder=1)
# accuracy_plot.set(ylim=(0, 1))
# plt.show()

fig, axs = plt.subplots(nrows=6, ncols=9)
vmin = np.min(confusion_matrices)
vmax = np.max(confusion_matrices)
row = 0
column = 0
fig.text(0.5, 0.96, 'Predicted Class', ha='center', va='center', size='xx-large')
fig.text(0.06, 0.5, 'Expected Class', ha='center', va='center', size='xx-large', rotation='vertical')
for confusion_matrix in confusion_matrices:
    labels = []
    for index, matrix_row in enumerate(confusion_matrix):
        row_sum = sum(matrix_row)
        labels.append([])
        for value in matrix_row:
            labels[index].append('{}\n{:.1f}%'.format(value, value / row_sum * 100))
    sns.heatmap(confusion_matrix, annot=labels, cbar=False, ax=axs[row, column], vmin=vmin, fmt='', cmap='inferno')
    column += 1
    if column > 8:
        row += 1
        column = 0

# fig.colorbar(axs[1].collections[0], cax=axs[2])

plt.show()
# sns.heatmap(a5, annot=True,  linewidths=.5)
# plt.show()
# visualize_csp_filters([(10,12)], [[1,1,a3, a4]], 2)
# plt.figure()
# plt.plot(a1, np.mean(a2, 0), label='Score')
# plt.axvline(0, linestyle='--', color='k', label='Onset')
# plt.axhline(0.5, linestyle='-', color='k', label='Chance')
# plt.xlabel('time (s)')
# plt.ylabel('classification accuracy')
# plt.title('Classification score over time')
# plt.legend(loc='lower right')
# plt.show()


# # b1, b2, _b3, _b4 = process('s14', chosen_bands2, channels)
# fig, (ax1, ax2) = plt.subplots(1)
# fig.suptitle('Vertically stacked subplots')
# ax1.plot(a1, np.mean(a2, 0))
# # ax2.plot(b1, np.mean(b2, 0))
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
