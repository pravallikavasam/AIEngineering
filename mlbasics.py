from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
X=[[2],
   [4],
   [6],
   [8]
   ]  #2Dimenion
y=[60000,80000,100000,120000]#1D, assigned each value for experience
model= LinearRegression()
model.fit(X,y)
###model.fit(X, y),"Model, learn the relationship between  Experience (X) and Salary (y)."
prediction=model.predict([[7]])
print(prediction)
print(model.coef_)
print(model.intercept_)

X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[50000,60000,70000,80000,90000,100000,110000,120000,130000,140000]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train:", X_train)
print("X_test:", X_test)
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(predictions)

mae=mean_absolute_error(y_test,predictions)
print("MAE:", mae)

#2nd model
X= [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y = [52000, 61000, 68000, 83000, 88000,
     103000, 108000, 123000, 127000, 142000]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Xtrain:",X_train)
print("ytrain:",y_train)
model.fit(X_train,y_train)
predictions2=model.predict(X_test)
print(predictions2)
mae2=mean_absolute_error(y_test,predictions2)
print("MAE:",mae2)
r2=r2_score(y_test,predictions2)
print("R2:",r2)