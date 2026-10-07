import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split


torch.manual_seed(42)

X= torch.tensor([
    [2.0,0.3],
    [4.0,0.5],
    [6.0,0.7],
    [8.0,0.9]

])

print(X)
print(X.shape)


y = torch.tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0]
])

print("y:")
print(y)

print("y shape:", y.shape)

model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()

)

print(model)

loss_function=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

#for epoch in range(3000):
#    predictions = model(X)
#    loss = loss_function(predictions,y)
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()

#    if epoch % 100 == 0:
#        print("Epoch:",epoch,"Loss:",loss.item())

#with torch.no_grad():
#    final_predictions = model(X)
#    predicted_classes=(final_predictions>0.5).float()

#print("final_predictions:")
#print(final_predictions)
#print("predicted_classes:")
#print(predicted_classes)

new_data = torch.tensor([
    [3.0, 0.4],
    [7.0, 0.8]
])

#with torch.no_grad():
#    new_predictions=model(new_data)
#    new_predicted_classes=(new_predictions>0.5).float()
#print("new_predictions:")
#print(new_predictions)
#print("new_predicted_classes:")
#print(new_predicted_classes)

print("Input data shape:", X.shape)
print("Output labels shape:", y.shape)

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.25,random_state=42)

print("Training_input_shape:",X_train.shape)
print("Validation_input_shape:",X_val.shape)
print("Training_labels_shape:",y_train.shape)
print("Validation_labels_shape:",y_val.shape)

for epoch in range(3000):
    predictions = model(X_train)
    #predictions = model(X)
    loss = loss_function(predictions,y_train)
    #loss = loss_function(predictions,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    val_predictions = model(X_val)
    val_loss = loss_function(val_predictions, y_val)

print("Validation predictions:")
print(val_predictions)
print("Validation loss:", val_loss.item())   
print("Actual validation label:")
print(y_val)
with torch.no_grad():
    val_predicted_class = (val_predictions >= 0.5).float()

print("Predicted validation class:")
print(val_predicted_class)

print("Validation input:")
print(X_val)

print("Validation actual label:")
print(y_val)
print("Training inputs:")
print(X_train)

print("Training labels:")
print(y_train)

print("Number of training records:", X_train.shape[0])
print("Number of validation records:", X_val.shape[0])

with torch.no_grad():
    train_predictions = model(X_train)
    train_predicted_classes = (train_predictions >= 0.5).float()

print("Training predictions:")
print(train_predictions)

print("Training predicted classes:")
print(train_predicted_classes)

print("Actual training labels:")
print(y_train)