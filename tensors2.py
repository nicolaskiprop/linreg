import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = torch.arange(0.0, 1.0, step=0.01)

X2 = torch.randint(2, size=(len(X),))

Y = (X * 0.7 + X2 * 0.2 - 0.3) + torch.normal(0, 0.1, size=(len(X),))


# Define the linear regression model
def linear_regression(x, weights, bias):
    return torch.matmul(x, weights) + bias

# Instantiate the model parameters
input_dim = 2
weights = torch.randn(input_dim, 1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

# Define the loss function
loss_fn = nn.MSELoss()

# Convert X and X2 to a single input tensor
X_combined = torch.stack([X, X2], dim=1)

# Convert Y to a column tensor
Y = Y.view(-1, 1)

# Experiment with different learning rates and epochs
learning_rates = [0.1, 0.01, 0.001]
num_epochs = [100, 500, 1000]
num_experiments = len(learning_rates)

# Lists to store the loss values and learning curves for each experiment
loss_values = []
learning_curves = []

# Training loop for each experiment
for i in range(num_experiments):
    # Instantiate the model parameters for each experiment
    weights = torch.randn(input_dim, 1, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)

    # Define the optimizer with the current learning rate
    optimizer = torch.optim.SGD([weights, bias], lr=learning_rates[i])

    # List to store the loss values for the current experiment
    experiment_losses = []

    for epoch in range(num_epochs[i]):
        # Forward pass
        outputs = linear_regression(X_combined, weights, bias)
        loss = loss_fn(outputs, Y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the loss value for the current epoch
        experiment_losses.append(loss.item())

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Experiment {i+1}/{num_experiments}, Epoch [{epoch+1}/{num_epochs[i]}], Loss: {experiment_losses[-1]:.4f}')

    # Store the loss values and learning curve for the current experiment
    loss_values.append(experiment_losses)
    learning_curves.append(list(range(1, num_epochs[i] + 1)))

# Plotting the learning curves for each experiment
plt.figure(figsize=(8, 5))

for i in range(num_experiments):
    plt.plot(learning_curves[i], loss_values[i], label=f'LR={learning_rates[i]}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
