import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data={
    "Experience":[2, 4, 6, np.nan, 10],
    "SkillScore" :[6, 7, np.nan, 8, 9],
    "Salary":[50000,65000,80000,90000,110000]
}
df=pd.DataFrame(data)
print(df)
print(df.isnull().sum())
print(df.dropna())
print(df["Experience"].mean())
print()
df["Experience"]=df["Experience"].fillna(df["Experience"].mean())
print(df["Experience"])

df["SkillScore"]=df["SkillScore"].fillna(df["SkillScore"].mean())
print(df)
print("check any nulls values left:")
print(df.isnull().sum())

df["Education"]=["Bachelors","Masters","Bachelors","PhD","Masters"]
print(df)
print(df.shape)

encoded_df=pd.get_dummies(df,columns=["Education"],dtype=int)
print(encoded_df)

scalar= StandardScaler()
features_to_scale=df[["Experience","SkillScore"]]
scaled_feature=scalar.fit_transform(features_to_scale)
print(scaled_feature)

from sklearn.model_selection import train_test_split

X = encoded_df.drop("Salary", axis=1)

y = encoded_df["Salary"]
print("X:")
print(X)
print("y:")
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("X_train:")
print(X_train)

print("X_test")
print(X_test)

scalar=StandardScaler()
X_train_scaled= scalar.fit_transform(X_train)
X_test_scaled=scalar.transform(X_test)

print("X_train_Scaled:")
print(X_train_scaled)

print("X_test_Scaled:")
print(X_test_scaled)
