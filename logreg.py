import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn.svm as svm
np.random.seed(12345)

n_points = 300
X = np.random.rand(n_points, 2)*10
X_test = np.random.rand(100, 2)*10
y = X[:,1] > X[:,0]

model1 = LogisticRegression().fit(X,y)
pred1 = model1.predict(X_test)

plt.figure(dpi=100)
plt.plot(X_test[:, 0][pred1 == False], X_test[:,1][pred1 == False], "r*")
plt.plot(X_test[:, 0][pred1 == True], X_test[:,1][pred1 == True], "b*")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")


y_circle = (X[:, 0] -5)**2 + (X[:, 1] -5)**2 < 4
model2 = LogisticRegression().fit(X, y_circle)
pred2 = model2.predict(X_test)

plt.figure(dpi=100)
plt.plot(X_test[:, 0][pred2 == False], X_test[:,1][pred2 == False], "r*")
plt.plot(X_test[:, 0][pred2 == True], X_test[:,1][pred2 == True], "b*")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")

model_svm = svm.SVC() .fit(X, y_circle)
pred_svm = model_svm.predict(X_test)
plt.figure(dpi=100)
plt.plot(X_test[:, 0][pred_svm == False], X_test[:,1][pred_svm == False], "r*")
plt.plot(X_test[:, 0][pred_svm == True], X_test[:,1][pred_svm == True], "b*")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
