import argparse
import copy
import sys
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torch.use_deterministic_algorithms(True)

INPUT_DIM = 24
DAY_SHIFT = 1 #set it to be between 1 and 10
HIDDEN_DIM = 1000
NUM_ITERS = 250
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1

class RNN(nn.Module):
    #Inspired by program from here https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
    def __init__(self, input_size, hidden_size=HIDDEN_DIM, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.linear1 = nn.Linear(input_size + hidden_size, output_size)
        self.linear2 = nn.Linear(input_size + hidden_size, hidden_size)
        self.outLinear = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, input, hidden):
        """print(input.dim(), hidden.dim())
        print(input.size(0), input.size(1))
        print(hidden.size(0), hidden.size(1))"""
        input_combined = torch.cat((input, hidden), 1)
        postLin = self.linear1(input_combined)
        hidden = self.linear2(input_combined)
        combined = torch.cat((hidden, postLin), 1)
        postRelu = F.relu(combined)
        postDrop = self.dropout(postRelu)
        output = self.outLinear(postDrop)
        #print(output)
        #output = self.softmax(output)
        #print(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train(X_data, y_data, model, optimizer):
    criterion = nn.MSELoss()
    X_data.unsqueeze_(-1)


    hidden = model.initHidden()

    #model.zero_grad()
    #model.requires_grad_(True)
    model.train() # Set model to "train mode",
    optimizer.zero_grad() 

    loss = torch.Tensor([0]) # you can also just simply use ``loss = 0``
    for i in range(X_data.size(0)):
        #print(f'i: {i}')
        input_tensor = X_data[i]  # Add batch dimension
        #print(input_tensor)
        #print(input_tensor.dim())
        if input_tensor.dim() > 2:
            input_tensor = input_tensor.squeeze()
            input_tensor = input_tensor.unsqueeze(1)

       
        input_tensor = input_tensor.permute(1,0)

        target_tensor = y_data[i]  # Add batch dimension
        # Forward pass
        output, hidden = model(input_tensor, hidden)

        # Compute loss for this iteration
        #print(output)
        output = output.squeeze()
        l = criterion(output, target_tensor)
        #print(output, target_tensor, l)

        """output, hidden = model(X_data[i], hidden)
        l = criterion(output, y_data[i])"""
        loss += l
    
    #print(loss)
    loss.backward()
    #print(loss)

    """for p in model.parameters():
        #print(p.data)
        print(p.grad.data)
        p.data.add_(p.grad.data, alpha=-LEARNING_RATE)
        #print(p.data)"""
    optimizer.step()

    return output, loss.item() / X_data.size(0), hidden
    #return output, loss.item(), hidden


def evaluate(model, X, y, name):
    """Evaluate the model using Mean Squared Error (MSE) for a regression task."""
    model.eval()  # Set model to "eval mode", turns off dropout and batch normalization, etc.

    with torch.no_grad():  # Context-manager that disables gradient calculation
        predictions = model(X)  # Get model predictions
        mse = nn.MSELoss()(predictions, y.view(-1, 1))  # Compute Mean Squared Error

        #mse = (predictions-y.view(-1,1))**2
        #mse = mse.mean()

    #print(f'    {name} MSE: {mse.item():.5f}')
    return mse.item()

def main():

    # Set random seed, for reproducibility
    torch.manual_seed(0)

    #read in the data
    price_data = np.loadtxt("updated_goldstock.csv", delimiter=",", dtype=float)

    #get the data that will be used for the solutions/predicted values
    solutions_data = price_data[10:]

    num_items = len(solutions_data)
    print("num items")
    print(num_items)


    y_train = solutions_data[:int(num_items * 0.7)]
    y_dev = solutions_data[int(num_items * 0.7):int(num_items * 0.8)]
    y_test = solutions_data[int(num_items * 0.8):] #int conversion not needed but used


    #get the input data
    cols_to_use = []
    for i in range(INPUT_DIM-1):
        cols_to_use.append(i + 1)
    input_data = np.loadtxt("combined_output.csv", delimiter=",", dtype=float, skiprows=1,usecols=cols_to_use)
    input_nums = input_data.shape[0]
    
    X_data = input_data[10-DAY_SHIFT:input_nums-DAY_SHIFT] #replace 10 with shift later
    gold_for_pred = price_data[9:-1]
    gold_column = gold_for_pred.reshape(-1, 1)

    X_data = np.concatenate((X_data, gold_column), axis=1)
    #X_data = gold_column
    input_dim = X_data.shape[1]

    #print(X_data)
    X_num = X_data.shape[0]

    print("num inputs")
    print(X_num)

    if X_num != num_items: #sanity check
        print("Error: Number of inputs does not match number of outputs")
        return

    X_train = X_data[:int(X_num * 0.7)]
    X_dev = X_data[int(X_num * 0.7):int(X_num * 0.8)]
    X_test = X_data[int(X_num * 0.8):]


    # Convert numpy arrays to PyTorch tensors and ensure they are of type float
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_dev = torch.tensor(X_dev, dtype=torch.float)
    y_dev = torch.tensor(y_dev, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    """X_train = X_train.unsqueeze(0)
    y_train = y_train.unsqueeze(0)
    X_dev = X_dev.unsqueeze(0)
    y_dev = y_dev.unsqueeze(0)
    X_test = X_test.unsqueeze(0)
    y_test = y_test.unsqueeze(0)"""

    print("Data Processed")
    #train model
    
    model = RNN(input_dim)


    print_every = 5
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every ``plot_every`` ``iters``

    start = time.time()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    hidden = 0
    loss = 0

    for iter in range(1, NUM_ITERS + 1):  
        output, loss, hidden = train(X_train, y_train, model, optimizer)
        total_loss += loss

        print(f'iter {iter}, training set loss {loss}')
    
    print(f'final training loss {loss}')


    #model has been trained, now try it on the dev set
    criterion = nn.MSELoss()
    X_dev.unsqueeze_(-1)
    dev_loss = 0 
    extra = 0

    for i in range(X_dev.size(0)):
        input_tensor = X_dev[i]  # Add batch dimension
        #print(input_tensor)
        #print(input_tensor.dim())
        if input_tensor.dim() > 2:
            input_tensor = input_tensor.squeeze()
            input_tensor = input_tensor.unsqueeze(1)

       
        input_tensor = input_tensor.permute(1,0)

        target_tensor = y_dev[i]  # Add batch dimension
        # Forward pass
        output, hidden = model(input_tensor, hidden)

        # Compute loss for this iteration
        #print(output)
        output = output.squeeze()
        #print(output, target_tensor)
        #print(i, ': ', output, target_tensor)
        extra += torch.abs(output - target_tensor)
        dev_loss += criterion(output, target_tensor)

    #print(f'dev loss: {dev_loss}')
    print(f'avg diff: {extra / X_dev.size(0)}')
    print(f'dev loss: {dev_loss / X_dev.size(0)}')


    #do the same for the test set
    X_test.unsqueeze_(-1)
    test_loss = 0 
    extra = 0

    output_values = []
    target_values = []
    differences = []
    for i in range(X_test.size(0)):
        input_tensor = X_test[i]  # Add batch dimension
        #print(input_tensor)
        #print(input_tensor.dim())
        if input_tensor.dim() > 2:
            input_tensor = input_tensor.squeeze()
            input_tensor = input_tensor.unsqueeze(1)

       
        input_tensor = input_tensor.permute(1,0)

        target_tensor = y_test[i]  # Add batch dimension
        # Forward pass
        output, hidden = model(input_tensor, hidden)

        # Compute loss for this iteration
        #print(output)
        output = output.squeeze()
        test_loss += criterion(output, target_tensor)
        extra += torch.abs(output - target_tensor)

        output_values.append(output.item())
        target_values.append(target_tensor.item())
        differences.append(torch.abs(output - target_tensor).item())


    #print(f'test loss: {test_loss}')
    print(f'avg diff: {extra / X_dev.size(0)}')
    print(f'test loss: {test_loss / X_test.size(0)}')

    plt.plot(output_values, label='Model Output')
    plt.plot(target_values, label='Real Value')
    plt.xlabel('Test Set Day')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Model Output vs. Real Value')
    plt.savefig('Results.png')
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(differences, label='Difference Between Real and Predicted', color='red')
    plt.title('Differences Between Real and Predicted Values')
    plt.xlabel('Test Set Day')
    plt.ylabel('Difference')
    plt.legend()
    plt.savefig('Differences.png')
    plt.show()


    #test sets to csv
    #np.savetxt("gold_test.csv", y_test.numpy(), delimiter=",")
    #print(model(X_test).detach().numpy())
    #np.savetxt("model_test.csv", model(X_test).detach().numpy(),delimiter=",")
    #np.savetxt("model_gold_test.csv", model(X_test).detach().numpy(),delimiter=",")

if __name__ == '__main__':
    main()
