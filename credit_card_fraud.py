import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
Df = pd.read_csv(r"C:\Users\HP\Desktop\datany\cd\creditcard.csv")
#Df.head()
non_fraud = len(Df[Df.Class == 0])
fraud_ = len(Df[Df.Class == 1])
fraud_pt = (fraud_/(fraud_+non_fraud))*100
print("the % of fraud data is ",fraud_pt)
scaler = StandardScaler()
Df["scaled_amount"] = scaler.fit_transform(Df["Amount"].values.reshape(-1,1))
Df = Df.drop(["Amount","Time"], axis = 1)
#Df.head()
X = Df.drop(["Class"], axis = 1)
Y = Df["Class"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)
md = RandomForestClassifier()
md.fit(x_train, y_train)
cls_pred = md.predict(x_test)
mdScore = md.score(x_test, y_test)
print(mdScore)
