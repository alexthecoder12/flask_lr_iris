import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import sys

iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

# Fit the model on training set
model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='auto')
model.fit(X_train, y_train)

# # save the model to disk
fname = 'model.sav'
joblib.dump(model, fname)

# # load the model from disk
loaded_model = joblib.load(fname)
result = loaded_model.score(X_test, y_test)
print('{:.2f}'.format(result))

print(loaded_model.predict(np.array([4.5, 1.5, 2.3, 1.0]).reshape(1, -1)))

sys.exit(1)
