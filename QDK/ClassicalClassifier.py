import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

h = .02  # step size in the mesh

with open('Data/MNIST/mnist_pca_30Components_3_6.json') as f:
  data = json.load(f)

trainingData = data["TrainingData"]
validationData = data["ValidationData"]


X_train = trainingData["Features"]
X_test = validationData["Features"]
Y_train = trainingData["Labels"]
Y_test = validationData["Labels"]

"""
X_train, X_test, Y_train, Y_test = \
        train_test_split(trainingData["Features"], trainingData["Labels"], test_size=.2, random_state=42)
"""

svc = SVC(gamma='auto')

svc.fit(X_train, Y_train)

score = svc.score(X_test, Y_test)

y_pred = svc.predict(X_test)

print(accuracy_score(Y_test, y_pred))
print(score)