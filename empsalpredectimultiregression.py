import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math
data={
    "Experience":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Education":[1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    "Skillscore":[55, 60, 62, 68, 70, 75, 78, 82, 88, 92],
    "salary":[45000, 48000, 55000, 59000, 63000, 72000, 76000, 81000, 87000, 93000]
}
df=pd.DataFrame(data)
print(df)
X=df[["Experience", "Education", "Skillscore"]]
y=df["salary"]

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

y_pred=model.predict(X_test)
print(y_pred)

print("Actalvalues:")
print(y_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=math.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2:",r2)

new_employee = pd.DataFrame({
    "Experience":[7],
    "Education":[2],
    "Skillscore":[80]
})

new_pred_salary=model.predict(new_employee)
print("New_Predictedsalary:",new_pred_salary)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


