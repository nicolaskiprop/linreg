import torch.nn as nn
import torch
import matplotlib.pyplot as plt

X = torch.arange(0.0, 1.0, step=0.01)
print(f'tensor X:{X}')

X2 = torch.randint(2, size=(len(X),))
print(f'tensor X2:{X2}')
Y = (X * 0.7 + X2 * 0.2 - 0.3) + torch.normal(0, 0.1, size=(len(X),))
print(f'tensor Y:{Y}')


#define linear regression model
def linear_regression(x, weight, bias):
    return torch.matmul(x, weight) + bias

#initialize and instantiate model params
input_size = 2
weights_untrained = torch.randn(input_size, 1)
bias_untrained = torch.randn(1)
weight = torch.randn(input_size, 1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

#define loss function
loss_fn = nn.MSELoss()

#define optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD([weight, bias], lr=learning_rate)

#convert x and x2 to single tensor
X_combined = torch.stack([X, X2], dim=1)
print(f'tensor X_combined:{X_combined}')

#convert y to a column vector
Y = Y.view(-1, 1)
print(f'tensor Y:{Y}')

#train model
num_epochs = 500
for epoch in range(num_epochs):
    #forward pass
    y_hat = linear_regression(X_combined, weight, bias)
    loss = loss_fn(y_hat, Y)
    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print loss after every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
    
#retrieve trained params
trained_weight = weight.detach()
trained_bias = bias.detach()
print('trained params')
print(f'weights: {trained_weight.squeeze().tolist()}')
print(f'bias: {trained_bias.squeeze().tolist()}')



# Plotting the trained and untrained parameters
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot([0, trained_weight[0].item()], [0, trained_weight[1].item()], label='Trained')
plt.plot([0, weights_untrained[0].item()], [0, weights_untrained[1].item()], label='Untrained')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('Weights')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Trained', 'Untrained'], [trained_bias.item(), bias_untrained.item()])
plt.ylabel('Bias')
plt.title('Bias')

plt.tight_layout()
plt.show()