import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
dir(digits)
df = pd.DataFrame(digits.data)
df["target"] = digits.target
X_train, X_test, y_train, y_test = train_test_split(df.drop(["target"],axis ="columns"),digits.target,test_size=.25)
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
modelScore = model.score(X_test, y_test)
print(modelScore)
