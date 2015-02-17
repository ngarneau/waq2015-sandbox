# coding: utf-8

import numpy as np

# Utilities
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file

# Premodel
from sklearn import preprocessing

# Model
from sklearn import neighbors

# Cross validation
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

# Metrics
from sklearn.metrics import classification_report

# Plot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Load necessary datasets
X, y = load_svmlight_file("../dataset/dataset")
X_mystery, y_mystery = load_svmlight_file("../dataset/mystery")

#-------
# Preprocess the data
# We can perform several preprocessing steps on our dataset.
# Here is one example of scaling the dataset.
#-------
print X.toarray()[0][:20]

scaler = preprocessing.StandardScaler(with_mean=False).fit(X)
X = scaler.transform(X)
X_mystery = scaler.transform(X_mystery)

print X.toarray()[0][:20]

#--------
# Grid search
# Search for the best hyper-parameters for our model.
# In our case we test for the number of neighbors and the weights picked by the algorithm.
#--------
tuned_parameters = [{'n_neighbors': [5, 9, 15, 25, 35, 45], 'weights': ['uniform', 'distance']}]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

def cross_val():
    scores = ['accuracy', 'f1']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

cross_val()


#--------
# Predict
# Use the whole dataset to train then predict the label on the mystery dataset
# Save the predicted values into svmlight format.
#--------
model = neighbors.KNeighborsClassifier(n_neighbors=35)
model.fit(X, y)
y_pred = model.predict(X_mystery)
dump_svmlight_file(X_mystery, y_pred, 'knn-35')

