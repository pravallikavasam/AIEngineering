import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import math
data = {
    "Experience":[1,2,3,4,5,6,7,8,9,10],
    "Salary":[45000,50000,55000,60000,65000,70000,75000,80000,85000,90000]
}
df=pd.DataFrame(data)
print(df)
X=df[["Experience"]]
y=df["Salary"]
print("X:")
print(X)
print("Y:")
print(y)
print(type(X))
print(type(y))
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
model=LinearRegression()
model.fit(X_train,y_train) #fit() means learn the relationship between Experience and Salary from the training data
y_pred=model.predict(X_test)
print("Predictedvalues:")
print(y_pred)
print("Actual_values:")
print(y_test)
mae=mean_absolute_error(y_test,y_pred)
rmse=math.sqrt(mae)
print("MAE:", mae)
print("RMSE:",rmse)
r2=r2_score(y_test,y_pred)
print("R2SCORE:",r2)
newdata=pd.DataFrame({
    "Experience": [12]
})
predictedsalary=model.predict(newdata)
print("Predictedsalary:",predictedsalary)

