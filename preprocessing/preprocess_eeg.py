import yaml
from os import listdir
from os.path import isfile, join
from preprocessing.preprocess_subject import preprocess_subject
import config


def is_matlab_file(file_name): return file_name.endswith('.mat')


def format_file(file_name): return "{}/{}".format(config.eeg_data_path, file_name)


filenames = [f for f in listdir(config.eeg_data_path) if isfile(join(config.eeg_data_path, f))]
subject_data_filenames = list(map(format_file, list(filter(is_matlab_file, filenames))))
preprocess_subject(subject_data_filenames)

