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
get_ipython().magic(u'matplotlib inline')


# Load necessary datasets
X, y = load_svmlight_file("../dataset/dataset")
X_mystery, y_mystery = load_svmlight_file("../dataset/mystery")


#-------
# Preprocess the data
# We can perform several preprocessing steps on our dataset.
# Here is one example of scaling the dataset.
# See also:
#    http://scikit-learn.org/stable/modules/preprocessing.html#normalization
#-------
print X.toarray()[0][:20]

X = preprocessing.StandardScaler().fit_transform(X.toarray())

X_mystery = preprocessing.StandardScaler().fit_transform(X_mystery.toarray())

print X[0][:20]


#--------
# Simple validation
# using train and test sets
#--------
model = neighbors.KNeighborsClassifier(n_neighbors=35)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
model.fit(X_train, y_train)
print model.score(X_test, y_test)


#--------
# Cross validation
#--------
model = neighbors.KNeighborsClassifier(n_neighbors=35, weights='distance')
scores = cross_validation.cross_val_score(model, X, y, verbose=3, cv=10)
print scores
print scores.mean()


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


# Predict using the best hyper parameter
model = neighbors.KNeighborsClassifier(n_neighbors=35)
model.fit(X, y)
y_pred = model.predict(X_mystery)

# Save the predicted labels
dump_svmlight_file(X_mystery, y_pred, 'ngarneau-scaled')


from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=35), n_estimators=100, max_samples=1.0, max_features=1.0)
scores = cross_validation.cross_val_score(bagging, X, y, verbose=3, cv=10)

print scores
print scores.mean()


bagging.fit(X, y)
y_pred = bagging.predict(X_mystery)

# Save the predicted labels
dump_svmlight_file(X_mystery, y_pred, 'bagging-knn-35')


