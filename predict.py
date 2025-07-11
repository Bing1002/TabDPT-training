from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTClassifier, TabDPTRegressor

# classification example
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTClassifier(path="./checkpoints/latest.ckpt")
model.fit(X_train, y_train)
y_pred = model.predict(X_test, temperature=0.8, context_size=1024, use_retrieval=True)
print("classification accuracy score = ", accuracy_score(y_test, y_pred))


# regression example
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTRegressor(path="./checkpoints/latest.ckpt")
model.fit(X_train, y_train)
y_pred = model.predict(X_test, context_size=512, use_retrieval=True)
print("regression r2 score =", r2_score(y_test, y_pred))
