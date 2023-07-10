import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Read the data
wine_data = pd.read_csv('wine.data')
#print(wine_data) 
X = wine_data.iloc[:, 1:].values
Y = wine_data.iloc[:, 0].values
#standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#convert to tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

input_dim = X.shape[1]
weights = torch.randn(input_dim, 1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
loss_fn = nn.MSELoss()
X_combined = X
Y = Y.view(-1, 1)
learning_rates = [0.1, 0.01, 0.001]
num_epochs = [100, 500, 1000]
num_experiments = len(learning_rates)
loss_values = []
learning_curves = []

model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

#training loop
for i in range(num_experiments):
    weights = torch.randn(input_dim, 1, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[i])
    experiment_losses = []

    for epoch in range(num_epochs[i]):
        #forward pass
        outputs = model(X_combined)
        loss = loss_fn(outputs, Y)

        #backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #store the loss value
        experiment_losses.append(loss.item())

        #print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {experiment_losses[-1]:.4f}')

    #store the loss values and learning curve for the current experiment
    loss_values.append(experiment_losses)
    learning_curves.append(list(range(1, num_epochs[i]+1)))
best_index = min(range(num_experiments), key=lambda i: loss_values[i][-1])

plt.plot(learning_curves[best_index], loss_values[best_index])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.grid(True)
plt.show()

