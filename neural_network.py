import torch 
import torch.nn as nn

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

for epoch in range(3000):
    predictions = model(X)
    loss = loss_function(predictions,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch:",epoch,"Loss:",loss.item())

with torch.no_grad():
    final_predictions = model(X)
    predicted_classes=(final_predictions>0.5).float()

print("final_predictions:")
print(final_predictions)
print("predicted_classes:")
print(predicted_classes)

new_data = torch.tensor([
    [3.0, 0.4],
    [7.0, 0.8]
])

with torch.no_grad():
    new_predictions=model(new_data)
    new_predicted_classes=(new_predictions>0.5).float()
print("new_predictions:")
print(new_predictions)
print("new_predicted_classes:")
print(new_predicted_classes)

