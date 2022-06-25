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
from sklearn.neural_network import MLPClassifier

from config import eeg_data_path
from data_classes.subject import Subject


def create_confusion_matrix(n_classes, predictions_sets, corrects_sets):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for fold_index in range(len(predictions_sets)):
        predictions = predictions_sets[fold_index]
        corrects = corrects_sets[fold_index]
        for index in range(len(predictions)):
            confusion_matrix[predictions[index]][corrects[index]] += 1
    return confusion_matrix


def process(subject_name, bands, selected_channels, reg=None, classifier='LDA'):
    tmin, tmax = 0., 2.
    event_id = dict(idle=0, left=1, right=2)
    subject = Subject(subject_name, eeg_data_path, True, True)

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

    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train[0])

    # Assemble a classifier
    if classifier == 'MLP':
        classifier = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=1,
                                   max_iter=1000)
    else:
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
    return w_times, scores_windows, csp, epochs[0].info, all_predictions, all_correct


def main(subjects_id, show_confusion_matrix=True):
    chosen_bands = [(9., 14.)]
    channels = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4']

    confusion_matrices = []
    all_scores = {'patient': [], 'accuracy': [], 'classifier': []}
    chances = {'values': [], 'patients': [],'classifier': []}
    for patient_index in subjects_id:
        print('Processing', patient_index, 'of', len(subjects_id), 'subjects')
        if patient_index < 10:
            patient_name = 's0{}'.format(patient_index)
        else:
            patient_name = 's{}'.format(patient_index)
        window_times, window_scores, csp_filters, epochs_info, predictions, corrects = process(patient_name,
                                                                                               chosen_bands, channels)
        if show_confusion_matrix:
            confusion_matrices.append(create_confusion_matrix(3, predictions, corrects))

        for i in range(len(predictions)):
            all_scores['patient'].append(patient_index)
            all_scores['accuracy'].append(accuracy_score(predictions[i], corrects[i]))
            all_scores['classifier'].append('LDA')
        chances['patients'].append(patient_index)
        chances['values'].append(st.binom.ppf(.9, len(predictions[0]), .3) / len(predictions[0]))
        window_times, window_scores, csp_filters, epochs_info, predictions, corrects = process(patient_name,
                                                                                               chosen_bands, channels,
                                                                                               classifier='MLP')
        if show_confusion_matrix:
            confusion_matrices.append(create_confusion_matrix(3, predictions, corrects))

        for i in range(len(predictions)):
            all_scores['patient'].append(patient_index)
            all_scores['accuracy'].append(accuracy_score(predictions[i], corrects[i]))
            all_scores['classifier'].append('MLP')
        # chances['patients'].append(patient_index)
        # chances['values'].append(st.binom.ppf(.9, len(predictions[0]), .3) / len(predictions[0]))

    all_scores = pd.DataFrame(data=all_scores)
    all_scores.to_csv('all_scores_model.csv')
    # chances = pd.DataFrame(chances['values'], index=chances['patients']).transpose()

    _accuracy_fig, accuracy_ax = plt.subplots()
    accuracy_plot = sns.barplot(x="patient", y="accuracy", hue='classifier', data=all_scores, ci="sd", ax=accuracy_ax, zorder=0)
    for i in range(len(chances['values'])):
        accuracy_ax.hlines(y=chances['values'][i], xmin=i - 0.4, xmax=i + 0.4, color='black')
    # sns.pointplot(data=chances, join=False, color='orange', ax=accuracy_ax, zorder=1)
    accuracy_plot.set(ylim=(0, 1))
    plt.show()
    if show_confusion_matrix:
        confusion_matrix_fig, confusion_matrix_ax = plt.subplots(nrows=6, ncols=9)
        row = 0
        column = 0
        confusion_matrix_fig.text(0.5, 0.96, 'Predicted Class', ha='center', va='center', size='xx-large')
        confusion_matrix_fig.text(0.06, 0.5, 'Expected Class', ha='center', va='center', size='xx-large',
                                  rotation='vertical')
        for subject_index, confusion_matrix in enumerate(confusion_matrices):
            labels = []
            for index, matrix_row in enumerate(confusion_matrix):
                row_sum = sum(matrix_row)
                labels.append([])
                for value in matrix_row:
                    labels[index].append('{}\n{:.1f}%'.format(value, value / row_sum * 100))
            sns.heatmap(confusion_matrix, annot=labels, cbar=False, ax=confusion_matrix_ax[row, column], fmt='',
                        cmap='inferno')
            confusion_matrix_ax[row, column].title.set_text('Subject: {}'.format(subject_index + 1))
            column += 1
            if column > 8:
                row += 1
                column = 0

        plt.show()


main(range(1, 53), show_confusion_matrix=False)
