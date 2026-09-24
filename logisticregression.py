import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data ={
    "Age": [22, 25, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58, 60, 62, 65],
    "MonthlySpend": [100, 120, 150, 180, 200, 220, 250, 270, 300, 320, 350, 380, 400, 420, 450],
    "SupportCalls": [1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6],
    "Churn": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
}
df=pd.DataFrame(data)
print(df)
X=df[["Age", "MonthlySpend", "SupportCalls"]]
y=df["Churn"]

print("X:")
print(X)
print("y:")
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("X_train:")
print(X_train)
print("X_test:")
print(X_test)

model= LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("y_pred:")
print(y_pred)
print("Actual:")
print(y_test)

probabilities=model.predict_proba(X_test)
print(probabilities)

