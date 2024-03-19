import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define your model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # Input size: 784, output size: 256
        self.fc2 = nn.Linear(256, 128)  # Input size: 256, output size: 128
        self.fc3 = nn.Linear(128, 10)  # Input size: 128, output size: 10 (number of classes)
        self.dropout = nn.Dropout(p=0.4)  # Dropout layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Second fully connected layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Final fully connected layer
        return x

def train_model(n_epochs, X_train, y_train, model, criterion, optimizer, batch_size=64):
    loss_over_time = []  # to track the loss as the network trains
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        # Mini-batch training without DataLoader
        for i in range(0, len(X_train), batch_size):
            inputs = torch.tensor(X_train[i:i+batch_size]).float()
            labels = torch.tensor(y_train[i:i+batch_size]).long()
            
            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            outputs = model(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass to calculate the parameter gradients
            loss.backward()

            # update the parameters
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to running_loss, we use .item()
            running_loss += loss.item()
            
            if (i / batch_size) % 100 == 99:  # print every 100 mini-batches
                avg_loss = running_loss / 100
                # record and print the avg loss over the mini-batches
                loss_over_time.append(avg_loss)
                print(f'Epoch: {epoch + 1}, Mini-batch: {i // batch_size + 1}, Avg. Loss: {avg_loss}')
                running_loss = 0.0

    print('Finished Training')
    return loss_over_time
