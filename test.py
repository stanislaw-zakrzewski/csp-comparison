import seaborn as sns
import matplotlib.pyplot as plt

oko = [[1, 3, 3], [4, 1, 2], [6, 0, 1]]
labels = []
for index, element in enumerate(oko):
    row_sum = sum(element)
    labels.append([])
    for value in element:
        labels[index].append('{}\n{:.1f}%'.format(value, value / row_sum * 100))
ax = sns.heatmap(oko, annot=labels, fmt="")

plt.show()
