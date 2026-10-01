import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text

data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 50, 55, 60],
    "Income": [30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000],
    "Buy": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df= pd.DataFrame(data)
print(df)

X = df[["Age","Income"]]
y = df["Buy"]

print("X:")
print(X)

print("y:")
print(y)

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

print("X_train:")
print(X_train)

print("X_test")
print(X_test)

model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train,y_train)

predections = model.predict(X_test)

print("Predicted:")
print(predections)

print("Actual:")
print(y_test)

accuracy = accuracy_score(y_test, predections)
print("Accuracy:", accuracy)

tree_rules = export_text(
    model,
    feature_names=["Age", "Income"]
)

print(tree_rules)



data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 50, 55, 60, 32, 38, 42, 48, 52],
    
    "Income": [30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000, 65000, 42000, 75000, 55000, 85000],
    
    "Buy": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
}

df= pd.DataFrame(data)
print(df)

X = df[["Age","Income"]]
y = df["Buy"]

print("X:")
print(X)

print("y:")
print(y)

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

print("X_train:")
print(X_train)

print("X_test")
print(X_test)

model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train,y_train)

predections = model.predict(X_test)

print("Predicted:")
print(predections)

print("Actual:")
print(y_test)

accuracy = accuracy_score(y_test, predections)
print("Accuracy:", accuracy)

tree_rules = export_text(
    model,
    feature_names=["Age", "Income"]
)

print(tree_rules)
train_predictions= model.predict(X_train)
print("Train_predictions:")
print(train_predictions)

train_accuracy=accuracy_score(y_train,train_predictions)
print("Test_Accuracy:", accuracy)
print("Train_Accuracy", train_accuracy)
