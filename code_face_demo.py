from sklearn.svm import SVC
import numpy as np

y1 = np.array([1, 2, 3, 4])
y2 = np.array([1, 2, 1, 2])
count = np.sum(y1 == y2)
print(count)

x = np.array([[0], [1], [2], [3]])
y = np.array([0, 1, 2, 3])

clf = SVC(decision_function_shape='ovo', gamma="scale")
test = np.array([[0], [2], [3], [7]])

