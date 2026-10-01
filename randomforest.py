import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 50, 55, 60, 32, 38, 42, 48, 52],

    "Income": [30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000, 65000, 42000, 75000, 55000, 85000],

    "Buy": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print(df)

X=df[["Age","Income"]]
y=df["Buy"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("X_train")
print(X_train)
print("X_test")
print(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

predictions=model.predict(X_test)
print("Predictions:")
print(predictions)

print("Actual:")
print(y_test)

accuracy=accuracy_score(y_test,predictions)
print("Accuracy:", accuracy)

train_predictions=model.predict(X_train)
print("Train_predictions:")
print(train_predictions)

train_accuracy=accuracy_score(y_train,train_predictions)
print("Train_Accuracy", train_accuracy)

precision = precision_score(y_test,predictions)
print("Precision", precision)

recall =recall_score(y_test,predictions)
print("Recall_Score:",recall)

F1Score = f1_score(y_test,predictions)
print("F1Score:",F1Score)

report=classification_report(y_test,predictions)
print("ClassificationReport:",report)

cv_scores=cross_val_score(model, X, y, cv=5)
print("Cross validation scores:")
print(cv_scores)

print("Average CV Accuracy:", cv_scores.mean())