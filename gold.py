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

#input dimension
INPUT_DIM = 24
DAY_SHIFT = 4

class TwoLayerMLP(nn.Module):
    #need to change hidden dim 
    def __init__(self, hidden_dim=200, dropout_prob=0.0):
        super(TwoLayerMLP, self).__init__()

        #create the linear layers
        self.linear1 = nn.Linear(INPUT_DIM, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.linearExtra = nn.Linear(hidden_dim, hidden_dim)

        #dropout layer
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        #linear
        firstLin = self.linear1(x)
        #relu
        postRelu = F.relu(firstLin)
        #add dropout
        postDrop = self.drop(postRelu)

        #postExtra = self.linearExtra(postDrop)
        #postRelu2 = F.relu(postExtra)

        #linear again
        output = self.linear2(postDrop)
        #output = self.linear2(postRelu2)
        return output

def train(model, X_train, y_train, X_dev, y_dev, lr=1e-10, batch_size=28, num_epochs=150):
    
    start_time = time.time()

    #loss function of MSE
    loss_func = nn.MSELoss()

    #optimiser
    optimizer = optim.SGD(model.parameters(), lr=lr)

    #pair up dataset
    train_dataset = [(X_train[i,:], y_train[i]) for i in range(len(y_train))]

    #for item in train_dataset:
    #    print(item)

     # Simple version of early stopping: save the best model checkpoint based on dev accuracy
    best_dev_acc = -1 
    best_checkpoint = None
    best_epoch = -1


    for t in range(num_epochs):
        #training loop
        model.train() # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.

        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
            # DataLoader automatically groups the data into batchse of roughly batch_size
                    # shuffle=True makes it so that the batches are randomly chosen in each epoch
            x_batch, y_batch = batch  
                    # unpack batch, which is a tuple (x_batch, y_batch)
                    # x_batch is tensor of size (B, D)
                    # y_batch is tensor of size (B,)
            optimizer.zero_grad() 
                    # Reset the gradients to zero
                    # Recall how backpropagation works---gradients are initialized to zero and then accumulated
                    # So we need to reset to zero before running on a new batch!
            logits = model(x_batch) 
            logits = logits.squeeze()
                    # tensor of size (B, C), each row is the logits (pre-softmax scores) for the C classes
                    # For MNIST, C=10
            loss = loss_func(logits, y_batch)
            #loss = ((logits-y_batch.view(-1,1))**2).mean()
            #print(loss)
                    # Compute the loss of the model output compared to true labels
            loss.backward() 
                    # Run backpropagation to compute gradients
            optimizer.step()
                    # Take a SGD step
                    # Note that when we created the optimizer, we passed in model.parameters()
                    # This is a list of all parameters of all layers of the model
                    # optimizer.step() iterates over this list and does an SGD update to each parameter

        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = evaluate(model, X_train, y_train, "Train")
        model.eval() 


        with torch.no_grad():
                    # Don't allocate memory for storing gradients, more efficient when not training
            dev_acc = evaluate(model, X_dev, y_dev, "Dev") 
            """if dev_acc < best_dev_acc:  
                # Save this checkpoint if it has best dev accuracy so far
                best_dev_acc = dev_acc
                best_checkpoint = copy.deepcopy(model.state_dict()) 
                best_epoch = t """
        print(f'Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}')


    # Set the model parameters to the best checkpoint across all epochs
    #model.load_state_dict(best_checkpoint)
    end_time = time.time() 
    print(f'Training took {end_time - start_time:.2f} seconds')
    #print(f'\nBest epoch was {best_epoch}, dev_acc={best_dev_acc:.5f}')

    #print(model(X_dev))
    #print(y_dev.view(-1, 1))
    #print(evaluate(model, X_dev, y_dev, "Dev"))

#def evaluate(model, X, y, name):
    #"""Measure and print accuracy of a predictor on a dataset."""
    """model.eval()  # Set model to "eval mode", e.g. turns dropout off if have dropout layers.

    with torch.no_grad():  # Don't allocate memory for storing gradients, more efficient when not training
        logits = model(X)  # tensor of size (N, 10)
        y_preds = torch.argmax(logits, dim=1)  # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
        acc = torch.mean((y_preds == y).float()).item()
    print(f'    {name} Accuracy: {acc:.5f}')
    return acc"""


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

    print("Data Processed")
    #train model
    model = TwoLayerMLP()
    model = model.float()
    print("Training Model:")
    train(model, X_train, y_train, X_dev, y_dev)

    #evaluate the model
    # Evaluate the model
    print('\nEvaluating final model:')
    train_acc = evaluate(model, X_train, y_train, 'Train')
    dev_acc = evaluate(model, X_dev, y_dev, 'Dev')
    test_acc = evaluate(model, X_test, y_test, 'Test')

    print(f"Train accuracy: {train_acc}\nDev Accuracy: {dev_acc}\nTest Accuracy: {test_acc}")

    
    #test sets to csv
    #np.savetxt("gold_test.csv", y_test.numpy(), delimiter=",")
    print(model(X_test).detach().numpy())
    #np.savetxt("model_test.csv", model(X_test).detach().numpy(),delimiter=",")
    #np.savetxt("model_gold_test.csv", model(X_test).detach().numpy(),delimiter=",")

if __name__ == '__main__':
    main()
