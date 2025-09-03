import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# data reading
data = pd.read_csv("student-mat.csv", sep = ";")

# data trimming
data = data[['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]

predict = "G3"

X = np.array(data.drop(predict, axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# model training
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

# saving the model
with open("student_model.pickle", "wb") as f:
    pickle.dump(linear f)

# display coefficients and intercepts
print ("Coefficients: \n", linear.coef_)
print ("Intercept: \n", linear.intercept_)

# peforming prediction of student grades
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])