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

torch.use_deterministic_algorithms(False)

INPUT_DIM = 24 #DO NOT TOUCH
DAY_SHIFT = 1  # set it to be between 1 and 10
HIDDEN_DIM = 20
NUM_ITERS = 30
LEARNING_RATE = 1e-1
DROPOUT_RATE = 0.4


class RNN(nn.Module):
    # copied from in class discussion section code
    def __init__(self, input_dim, num_layers=10, hidden_dim=HIDDEN_DIM, dropout_prob=DROPOUT_RATE):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # Predicting a single output value
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension

            # Initialize the hidden state
        batch_size = x.size(0)  # Adapt batch size from input
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(x.device)

        # Forward pass through RNN
        out, hn = self.rnn(x, h0)
        relu = nn.ReLU()

        # out has shape (batch_size, sequence_length, hidden_dim)
        # Process output from each time step
        outputs = []
        for t in range(out.size(1)):  # loop over time steps
            out_t = self.fc(out[:, t, :])
            out_t = relu(out_t)  # Apply the first FC layer to each time step output
            out_t = self.linear2(out_t)  # Apply the second FC layer to get the final prediction
            outputs.append(out_t)

        # Stack outputs to match the input sequence structure
        outputs = torch.stack(outputs, dim=1)
        return outputs


# for each time step input will be (data points, hidden state) and the outputs will be (next price, next hidden state)
def train(model, X_train, y_train, X_dev, y_dev, lr=LEARNING_RATE, num_iters=NUM_ITERS):
    start_time = time.time()

    # loss function of MSE
    loss_func = nn.MSELoss()

    # optimiser
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for t in range(num_iters):
        # training loop
        model.train()  # Set model to "train mode",
        optimizer.zero_grad()
        # Reset the gradients to zero
        # Recall how backpropagation works---gradients are initialized to zero and then accumulated

        # Forward pass using the whole training dataset
        logits = model(X_train)
        logits = logits.squeeze()  # Adjust the output dimensions if necessary
        loss = loss_func(logits, y_train)
        # Compute the loss of the model output compared to true labels
        loss.backward()
        # Run backpropagation to compute gradients
        optimizer.step()

        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = evaluate(model, X_train, y_train, "Train")
        dev_acc = evaluate(model, X_dev, y_dev, "Dev")

        print(f'Iteration {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}')

    # Set the model parameters to the best checkpoint across all epochs
    end_time = time.time()
    print(f'Training took {end_time - start_time:.2f} seconds')

    # print(model(X_dev))
    # print(y_dev.view(-1, 1))
    # print(evaluate(model, X_dev, y_dev, "Dev"))


def evaluate(model, X, y, name):
    """Evaluate the model using Mean Squared Error (MSE) for a regression task."""
    model.eval()  # Set model to "eval mode", turns off dropout and batch normalization, etc.

    with torch.no_grad():  # Context-manager that disables gradient calculation
        predictions = model(X)  # Get model predictions
        mse = nn.MSELoss()(predictions, y.view(-1, 1))  # Compute Mean Squared Error

        # mse = (predictions-y.view(-1,1))**2
        # mse = mse.mean()

    # print(f'    {name} MSE: {mse.item():.5f}')
    return mse.item()

def plot_results(y_test, predictions):
    # Convert tensors to numpy arrays if they are not already
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()

    # Squeeze unnecessary dimensions
    y_test = np.squeeze(y_test)
    predictions = np.squeeze(predictions)

    # Plotting Real vs Predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Real Values')
    plt.plot(predictions, label='Predicted Values', linestyle='--')
    plt.title('Model Output vs. Real Output')
    plt.xlabel('Test Set Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plotting the differences
    plt.figure(figsize=(10, 5))
    differences = np.abs(y_test - predictions)
    plt.plot(differences, label='Difference Between Real and Predicted', color='red')
    plt.title('Differences Between Real and Predicted Values')
    plt.xlabel('Test Set Day')
    plt.ylabel('Difference')
    plt.legend()
    plt.show()



def main():
    # Set random seed, for reproducibility
    torch.manual_seed(0)

    # read in the data
    price_data = np.loadtxt("updated_goldstock.csv", delimiter=",", dtype=float)

    # get the data that will be used for the solutions/predicted values
    solutions_data = price_data[10:]

    num_items = len(solutions_data)
    print("num items")
    print(num_items)

    y_train = solutions_data[:int(num_items * 0.7)]
    y_dev = solutions_data[int(num_items * 0.7):int(num_items * 0.8)]
    y_test = solutions_data[int(num_items * 0.8):int(num_items * 1)]  # int conversion not needed but used

    # get the input data
    cols_to_use = []
    for i in range(INPUT_DIM - 1):
        cols_to_use.append(i + 1)
    input_data = np.loadtxt("combined_output.csv", delimiter=",", dtype=float, skiprows=1, usecols=cols_to_use)
    input_nums = input_data.shape[0]

    X_data = input_data[10 - DAY_SHIFT:input_nums - DAY_SHIFT]  # replace 10 with shift later
    gold_for_pred = price_data[9:-1]
    gold_column = gold_for_pred.reshape(-1, 1)

    X_data = np.concatenate((X_data, gold_column), axis=1)
    # X_data = gold_column
    input_dim = X_data.shape[1]

    # print(X_data)
    X_num = X_data.shape[0]

    print("num inputs")
    print(X_num)

    if X_num != num_items:  # sanity check
        print("Error: Number of inputs does not match number of outputs")
        return

    X_train = X_data[:int(num_items * 0.7)]
    X_dev = X_data[int(num_items * 0.7):int(num_items * 0.8)]
    X_test = X_data[int(num_items * 0.8):int(num_items * 1)]

    # Convert numpy arrays to PyTorch tensors and ensure they are of type float
    X_train = torch.tensor(X_train, dtype=torch.float)
    # X_train = X_train.reshape(-1, int(X_num * 0.7), INPUT_DIM)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_dev = torch.tensor(X_dev, dtype=torch.float)
    y_dev = torch.tensor(y_dev, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    print("Data Processed")
    # train model
    model = RNN(input_dim)

    print("Training Model:")
    train(model, X_train, y_train, X_dev, y_dev)

    # evaluate the model
    # Evaluate the model
    print('\nEvaluating final model:')
    train_acc = evaluate(model, X_train, y_train, 'Train')
    dev_acc = evaluate(model, X_dev, y_dev, 'Dev')
    test_acc = evaluate(model, X_test, y_test, 'Test')

    print(f"Train accuracy: {train_acc}\nDev Accuracy: {dev_acc}\nTest Accuracy: {test_acc}")

    # test sets to csv
    # np.savetxt("gold_test.csv", y_test.numpy(), delimiter=",")
    # print(model(X_test).detach().numpy())
    # np.savetxt("model_test.csv", model(X_test).detach().numpy(),delimiter=",")
    # np.savetxt("model_gold_test.csv", model(X_test).detach().numpy(),delimiter=",")
    predictions = model(X_test)
    print(predictions)

    # Call the plotting function

    plot_results(y_test, predictions)


if __name__ == '__main__':
    main()