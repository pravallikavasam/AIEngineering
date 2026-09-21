from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
X=[[1],[2],[3],[5],[6],[7]]
y=[0,0,0,1,1,1]
model=LogisticRegression()
model.fit(X,y)
prediction=model.predict([[4]])
print(prediction)
probability=model.predict_proba([[4]])
print(probability)
print(model.predict_proba([[6]]))

X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[0,0,0,0,0,1,1,1,1,1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train:",X_train)
print("X_test:",X_test)
model=LogisticRegression()
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print("predection1:",prediction1)
print("actualdata:",y_test)
accuracy=accuracy_score(y_test,prediction1)
print("Accuracy:",accuracy)
cm=confusion_matrix(y_test,prediction1)
print("Confusion Matrix:")
print(cm)

precision= precision_score(y_test,prediction1)
recall=recall_score(y_test,prediction1)

print("Precision:",precision)
print("Recall:", recall)
f1= f1_score(y_test,prediction1)
print("f1score:",f1)