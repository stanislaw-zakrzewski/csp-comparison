import numpy as np
import matplotlib.pyplot as plt

from mne import filter

SUBJECT_NAME = 's14'

data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(SUBJECT_NAME))

sampling_frequency = 512

start = int(0 * sampling_frequency)
end = int(7 * sampling_frequency)

sig1 = list(map(float, data_left[49][start:end]))
sig2 = list(map(float, data_left[12][start:end]))

filtered1 = filter.filter_data(sig1, sampling_frequency, 10, 12)
filtered2 = filter.filter_data(sig1, sampling_frequency, 20, 24)

filtered3 = filter.filter_data(sig2, sampling_frequency, 10, 12)
filtered4 = filter.filter_data(sig2, sampling_frequency, 20, 24)

squared1 = []
squared2 = []
squared3 = []
squared4 = []



for s in range(len(sig1)):
    squared1.append(np.abs(filtered1[s]))
    squared2.append(np.abs(filtered2[s]))
    squared3.append(np.abs(filtered3[s]))
    squared4.append(np.abs(filtered4[s]))

avg1 = []
avg2 = []
avg3 = []
avg4 = []

N = 512
for i in range(len(squared1)):
    avg1.append(np.average(squared1[max(0, i - N): min(end, i+N)]))
    avg2.append(np.average(squared2[max(0, i - N): min(end, i+N)]))
    avg3.append(np.average(squared3[max(0, i - N): min(end, i+N)]))
    avg4.append(np.average(squared4[max(0, i - N): min(end, i+N)]))

# plt.plot(sig1)
# plt.plot(sig2)
plt.plot(avg1, label='C4 10Hz-12Hz')
plt.plot(avg2, label='C4 20Hz-24Hz')
plt.plot(avg3, label='C3 10Hz-12Hz')
plt.plot(avg4, label='C3 20Hz-24Hz')
plt.axvline(2*sampling_frequency)
plt.axvline(5*sampling_frequency)
plt.legend(loc="upper left")
plt.show()
