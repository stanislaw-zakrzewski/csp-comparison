import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from mne import Epochs, pick_types
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
import timeit

from config import eeg_data_path
from data_classes.subject import Subject


def get_individual_accuracy(predicions, correct):
    all = [0, 0, 0]
    cor = [0, 0, 0]
    for i in range(len(predicions)):
        all[correct[i]] += 1
        if predicions[i] == correct[i]:
            cor[predicions[i]] += 1
    results = [2, 2, 2]
    if all[0] > 0: results[0] = cor[0] / all[0]
    if all[1] > 0: results[1] = cor[1] / all[1]
    if all[2] > 0: results[2] = cor[2] / all[2]
    return results


def create_confusion_matrix(n_classes, predictions_sets, corrects_sets):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for fold_index in range(len(predictions_sets)):
        predictions = predictions_sets[fold_index]
        corrects = corrects_sets[fold_index]
        for index in range(len(predictions)):
            confusion_matrix[predictions[index]][corrects[index]] += 1
    return confusion_matrix


def process(subject_name, bands, selected_channels, reg=None, balanced=True):
    runtime = []
    tmin, tmax = 0., 2.
    event_id = dict(idle=0, left=1, right=2)
    subject = Subject(subject_name, eeg_data_path, True, balanced)

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
        epochs.append(
            Epochs(filtered_raw_signals[index], subject.events, event_id, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True))
        epochs_train.append(epochs[index].copy().crop(tmin=0., tmax=2.))

        epochs_data.append(epochs[index].get_data())
        epochs_data_train.append(epochs_train[index].get_data())
    labels = np.array(epochs[0].events[:, -1])

    cv = ShuffleSplit(1, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train[0])

    # Assemble a classifier
    # classifier = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=1,
    #                            max_iter=1000)  # Originally: LinearDiscriminantAnalysis()
    classifier = LinearDiscriminantAnalysis()
    csp_n_components = 10 if len(selected_channels) == 0 else min(len(selected_channels), 10)
    csp = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)

    sfreq = raw_signals[0].info['sfreq']
    w_length = int(sfreq)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data[0].shape[2] - w_length, w_step)

    scores_windows = []
    all_predictions = []
    all_correct = []
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = []
        x_test_csp = []
        start_train = timeit.default_timer()
        for edt in epochs_data_train:
            if len(x_train_csp) > 0:
                x_train_csp = np.concatenate((x_train_csp, csp.fit_transform(edt[train_idx], y_train)), axis=1)
                x_test_csp = np.concatenate((x_test_csp, csp.transform(edt[test_idx])), axis=1)
            else:
                x_train_csp = csp.fit_transform(edt[train_idx], y_train)
                x_test_csp = csp.transform(edt[test_idx])

        # fit classifier
        classifier.fit(x_train_csp, y_train)

        stop_train = timeit.default_timer()
        runtime.append(stop_train-start_train)

        start_test = timeit.default_timer()
        X_test_csp = csp.transform(epochs_data[0][test_idx])
        predictions = classifier.predict(X_test_csp)
        stop_test = timeit.default_timer()
        runtime.append(stop_test - start_test)
        all_predictions.append(predictions)
        all_correct.append(y_test)

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

    w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin
    return w_times, scores_windows, csp, epochs[0].info, all_predictions, all_correct, runtime


def process_hierarchical(subject_name, bands, selected_channels, reg=None):
    runtime = []
    tmin, tmax = 0., 2.
    event_1_id = dict(idle=0, imagery_movement=1)
    event_2_id = dict(left=1, right=2)
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
    epochs_1 = []
    epochs_1_train = []
    epochs_1_data = []
    epochs_1_data_train = []
    epochs_2 = []
    epochs_2_train = []
    epochs_2_data = []
    epochs_2_data_train = []

    # Apply band-pass filter
    for index, band in enumerate(bands):
        filtered_raw_signals.append(
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=.5, h_trans_bandwidth=.5, fir_design='firwin',
                                      skip_by_annotation='edge'))

    picks = pick_types(filtered_raw_signals[0].info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    all_labels = np.array([])
    for index, band in enumerate(bands):
        events_1 = []
        events_2 = []
        for i in subject.events:
            all_labels = np.append(all_labels, int(i[2]))
            if i[2] != 0:
                events_2.append(i)
            if i[2] == 2:
                events_1.append([i[0], i[1], 1])
            else:
                events_1.append(i)
        epochs_1.append(
            Epochs(filtered_raw_signals[index], events_1, event_1_id, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True))
        epochs_2.append(
            Epochs(filtered_raw_signals[index], events_2, event_2_id, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True))
        epochs_1_train.append(epochs_1[index].copy().crop(tmin=0., tmax=2.))
        epochs_2_train.append(epochs_2[index].copy().crop(tmin=0., tmax=2.))

        epochs_1_data.append(epochs_1[index].get_data())
        epochs_2_data.append(epochs_2[index].get_data())
        epochs_1_data_train.append(epochs_1_train[index].get_data())
        epochs_2_data_train.append(epochs_2_train[index].get_data())
    labels_1 = np.array(epochs_1[0].events[:, -1])
    labels_2 = np.array(epochs_2[0].events[:, -1])

    cv = ShuffleSplit(1, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_1_data_train[0])

    # Assemble a classifier
    # classifier = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=1,
    #                            max_iter=1000)  # Originally: LinearDiscriminantAnalysis()
    classifier_1 = LinearDiscriminantAnalysis()
    classifier_2 = LinearDiscriminantAnalysis()
    csp_n_components = 10 if len(selected_channels) == 0 else min(len(selected_channels), 10)
    csp_1 = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)
    csp_2 = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)

    sfreq = raw_signals[0].info['sfreq']
    w_length = int(sfreq)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_1_data[0].shape[2] - w_length, w_step)

    scores_windows = []
    all_predictions = []
    all_correct = []
    for train_idx_1, test_idx_1 in cv_split:
        train_idx_2 = []
        test_idx_2 = []
        y_train_1, y_test_1 = labels_1[train_idx_1], labels_1[test_idx_1]

        for index, value in enumerate(train_idx_1):
            if y_train_1[index] != 0:
                train_idx_2.append(int((value - 1) / 2))
        for index, value in enumerate(test_idx_1):
            if y_test_1[index] != 0:
                test_idx_2.append(int((value - 1) / 2))
        y_train_2, y_test_2 = labels_2[train_idx_2], labels_2[test_idx_2]
        start_train = timeit.default_timer()
        x_train_csp_1 = []
        x_test_csp_1 = []
        for edt in epochs_1_data_train:
            if len(x_train_csp_1) > 0:
                x_train_csp_1 = np.concatenate((x_train_csp_1, csp_1.fit_transform(edt[train_idx_1], y_train_1)),
                                               axis=1)
                x_test_csp_1 = np.concatenate((x_test_csp_1, csp_1.transform(edt[test_idx_1])), axis=1)
            else:
                x_train_csp_1 = csp_1.fit_transform(edt[train_idx_1], y_train_1)
                x_test_csp_1 = csp_1.transform(edt[test_idx_1])

        x_train_csp_2 = []
        x_test_csp_2 = []
        for edt in epochs_2_data_train:
            if len(x_train_csp_2) > 0:
                x_train_csp_2 = np.concatenate((x_train_csp_2, csp_2.fit_transform(edt[train_idx_2], y_train_2)),
                                               axis=1)
                x_test_csp_2 = np.concatenate((x_test_csp_2, csp_2.transform(edt[test_idx_2])), axis=1)
            else:
                x_train_csp_2 = csp_2.fit_transform(edt[train_idx_2], y_train_2)
                x_test_csp_2 = csp_2.transform(edt[test_idx_2])

        # fit classifier
        classifier_1.fit(x_train_csp_1, y_train_1)
        classifier_2.fit(x_train_csp_2, y_train_2)
        stop_train = timeit.default_timer()
        runtime.append(stop_train - start_train)

        start_test = timeit.default_timer()
        X_test_csp_1 = csp_1.transform(epochs_1_data[0][test_idx_1])
        predictions = classifier_1.predict(X_test_csp_1)
        pred = []
        for index, value in enumerate(test_idx_1):
            if predictions[index] == 0:
                pred.append(0)
            else:

                csp_value = csp_2.transform(epochs_1_data[0][[value]])
                classifier_value = classifier_2.predict(csp_value)
                pred.append(classifier_value[0])

        stop_test = timeit.default_timer()
        runtime.append(stop_test - start_test)

        all_predictions.append(pred)
        all_correct.append(list(map(int, all_labels[test_idx_1])))

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            test_split_data = []
            for edt in epochs_1_data:
                if len(test_split_data) > 0:
                    test_split_data = np.concatenate(
                        (test_split_data, edt[test_idx_1][:, :, n:(n + w_length)]),
                        axis=1)
                else:
                    test_split_data = edt[test_idx_1][:, :, n:(n + w_length)]
            csp_1_test_res = csp_1.transform(test_split_data)

            step_1_predicions = classifier_1.predict(csp_1_test_res)
            step_predictions = []
            for index, value in enumerate(step_1_predicions):
                if value == 0:
                    step_predictions.append(value)
                else:
                    csp_2_value = csp_2.transform(np.array([test_split_data[index]]))
                    step_2_prediction = classifier_2.predict(csp_2_value)
                    step_predictions.append(step_2_prediction[0])
            score = accuracy_score(all_labels[test_idx_1], step_predictions)
            score_this_window.append(score)

    w_times = (w_start + w_length / 2.) / sfreq + epochs_1[0].tmin
    return w_times, scores_windows, csp_1, epochs_1[0].info, all_predictions, all_correct, runtime


def main(subjects_id):
    chosen_bands_1 = [(6., 30.)]
    chosen_bands_2 = [(9., 14.)]
    chosen_bands_3 = [(10., 12.)]
    channels_1 = ['C3', 'C4']
    channels_2 = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4']
    channels_3 = ['FC5', 'FC3', 'FC2', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1', 'FC2', 'FC4', 'FC6', 'C2', 'C4', 'C6',
                  'CP2', 'CP4', 'CP6']
    channels_4 = []

    confusion_matrices = []
    times = {'patient': [], 'channels': [], 'train': [], 'test': []}
    chances = {'values': [], 'patients': []}
    for patient_index in subjects_id:
        print('Processing', patient_index, 'of', len(subjects_id), 'subjects')
        if patient_index < 10:
            patient_name = 's0{}'.format(patient_index)
        else:
            patient_name = 's{}'.format(patient_index)
        window_times, window_scores, csp_filters, epochs_info, predictions, corrects, runtime = process(patient_name,
                                                                                               chosen_bands_2, channels_1)

        times['patient'].append(patient_index)
        times['channels'].append(2)
        times['train'].append(runtime[0])
        times['test'].append(runtime[1])


        window_times, window_scores, csp_filters, epochs_info, predictions, corrects, runtime = process(patient_name,
                                                                                               chosen_bands_2, channels_2,
                                                                                               )
        times['patient'].append(patient_index)
        times['channels'].append(10)
        times['train'].append(runtime[0])
        times['test'].append(runtime[1])

        window_times, window_scores, csp_filters, epochs_info, predictions, corrects, runtime = process(
            patient_name,
            chosen_bands_2, channels_3)
        times['patient'].append(patient_index)
        times['channels'].append(18)
        times['train'].append(runtime[0])
        times['test'].append(runtime[1])

        window_times, window_scores, csp_filters, epochs_info, predictions, corrects, runtime = process(
            patient_name,
            chosen_bands_2, channels_4)
        times['patient'].append(patient_index)
        times['channels'].append(64)
        times['train'].append(runtime[0])
        times['test'].append(runtime[1])

    times = pd.DataFrame(data=times)
    times.to_csv('times_channels.csv')

    # all_scores = pd.DataFrame(data=all_scores)
    # individual_classes = pd.DataFrame(data=individual_classes)
    # all_scores.to_csv('all_scores_channels.csv')
    # individual_classes.to_csv('individual_classes_channels.csv')
    # chances = pd.DataFrame(chances['values'], index=chances['patients']).transpose()
    #
    # _accuracy_fig, accuracy_ax = plt.subplots()
    # accuracy_plot = sns.barplot(x="patient", y="accuracy", hue='bands', data=all_scores, ci="sd", ax=accuracy_ax,
    #                             zorder=0)
    # # sns.pointplot(data=chances, join=False, color='orange', ax=accuracy_ax, zorder=1)
    # accuracy_plot.set(ylim=(0, 1))
    # plt.show()
    #
    # confusion_matrix_fig, confusion_matrix_ax = plt.subplots(nrows=6, ncols=9)
    # row = 0
    # column = 0
    # confusion_matrix_fig.text(0.5, 0.96, 'Predicted Class', ha='center', va='center', size='xx-large')
    # confusion_matrix_fig.text(0.06, 0.5, 'Expected Class', ha='center', va='center', size='xx-large',
    #                           rotation='vertical')
    # for subject_index, confusion_matrix in enumerate(confusion_matrices):
    #     labels = []
    #     for index, matrix_row in enumerate(confusion_matrix):
    #         row_sum = sum(matrix_row)
    #         labels.append([])
    #         for value in matrix_row:
    #             labels[index].append('{}\n{:.1f}%'.format(value, value / row_sum * 100))
    #     sns.heatmap(confusion_matrix, annot=labels, cbar=False, ax=confusion_matrix_ax[row, column], fmt='',
    #                 cmap='inferno')
    #     confusion_matrix_ax[row, column].title.set_text('Subject: {}'.format(subject_index + 1))
    #     column += 1
    #     if column > 8:
    #         row += 1
    #         column = 0
    #
    # plt.show()


main(range(1, 53))
