from sklearn import neighbors
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import helper
import joblib


az_training, az_lables = helper.load_az_dataset()
mnist_training, mnist_lables = helper.load_minst_dataset()


data = np.concatenate([az_training, mnist_training])
lables = np.hstack([az_lables, mnist_lables])


training_x, test_x, training_Y, test_Y = train_test_split(
    data, lables, test_size=0.33, random_state=42, shuffle=True
)



# param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf', 'linear']}

param_grid = dict(n_neighbors=list(range(1, 31)))
model = neighbors.KNeighborsClassifier()
grid_search = GridSearchCV(model, param_grid, cv=10)
model = grid_search.fit(training_x, training_Y)



predictions = model.predict(test_x)

print(predictions)
print("Accuracy: ", accuracy_score(test_Y, predictions))
print(classification_report(test_Y, predictions))

joblib.dump(model, "model/emnist_byclass_svm_sos.pkl")