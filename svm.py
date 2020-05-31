import numpy as np
import matplotlib.pyplot as plt

import sklearn.linear_model as lm
import sklearn.svm as svm
np.random.seed(12345)
n_points = 300
X = np.random.rand(n_points, 2)*10
y_circle = (X[:, 0] -5)**2 + (X[:, 1] -5)**2 < 4
X_test = np.random.rand(15000, 2)*10
model_svm = svm.SVC() .fit(X, y_circle)

pred_svm = model_svm.predict(X_test)
plt.figure(dpi=100)
plt.plot(X_test[:, 0][pred_svm == False], X_test[:,1][pred_svm == False], "r*")
plt.plot(X_test[:, 0][pred_svm == True], X_test[:,1][pred_svm == True], "b*")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()