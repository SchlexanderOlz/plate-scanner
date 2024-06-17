from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib


training_data = pd.read_csv("data/emnist-byclass-train.csv", header=None, nrows=10000) 
training_Y = training_data.iloc[:, 0]
training_x = training_data.iloc[:, 1:]


# training_x = scaler = preprocessing.StandardScaler().fit_transform(training_x)

param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf', 'linear']}


model = svm.SVC(kernel='rbf', C=1, gamma=0.01)
grid_search = GridSearchCV(model, param_grid, cv=5)
model = grid_search.fit(training_x, training_Y)



test_data = pd.read_csv("data/emnist-byclass-test.csv", header=None, nrows=1000)

test_Y = test_data.iloc[:, 0]
test_x = test_data.iloc[:, 1:]

predictions = model.predict(test_x)

print(predictions)
print("Accuracy: ", np.mean(predictions == test_Y))
print("Accuracy: ", accuracy_score(test_Y, predictions))
print(classification_report(test_Y, predictions))

joblib.dump(model, "model/emnist_byclass_svm.pkl")