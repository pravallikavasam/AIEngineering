from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
X = [
    [1, 1, 5],
    [2, 1, 6],
    [3, 2, 6],
    [4, 2, 7],
    [5, 2, 7],
    [6, 2, 8],
    [7, 3, 8],
    [8, 3, 9],
    [9, 3, 9],
    [10, 3, 10]
]

y = [
    45000, 50000, 60000, 68000, 75000,
    82000, 90000, 100000, 110000, 120000
]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model= LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("Predicted:", predictions)
print("Actual:", y_test)
mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)
r2 = r2_score(y_test, predictions)

print("R2:", r2)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
new_person = [[12, 3, 9]]

predicted_salary = model.predict(new_person)

print("Predicted Salary:", predicted_salary)


import pandas as pd

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Education":  [1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    "SkillScore": [5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
    "Salary":     [45000, 50000, 60000, 68000, 75000, 82000, 90000, 100000, 110000, 120000]
}

df = pd.DataFrame(data)
print(df)

X= df[["Experience","Education","SkillScore"]]
y=df["Salary"]
print("X:")
print(X)
print("y:")
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
model=LinearRegression()
model.fit(X_train,y_train)

predic=model.predict(X_test)

print("Prediction:")
print(predic)

print("Actual:")
print(y_test)

mae=mean_absolute_error(y_test,predic)
print("MAE:",mae)
r2sc=r2_score(y_test,predic)

print("r2score:")
print(r2sc)