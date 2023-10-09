# import all necessary libraries
import numpy as np  # For matrices and MATLAB like functions
import random
from sklearn.model_selection import (
    train_test_split,
)  # To split data into train and test set

# for plotting graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_california_housing
import pickle

Viz_Data = True


def set_mean(X):
    # Write your function here
    return np.mean(X, axis=0, keepdims=True)


def set_standardDeviation(X):
    # Write your function here
    return np.std(X, axis=0, keepdims=True)


def normalize(X, Me, St):
    # Write your function here
    return (X - Me) / St

# this function appends a column of ones to X (bias column)
def add_bias_column(X):
    bias = np.ones((X.shape[0],1))
    X = np.concatenate((X, bias), axis=1)
    return X

def plot_predictions(testY, y_pred, output_type):
    # Write your function here
    plt.figure(figsize=(15,8))
    plt.plot(testY.squeeze(), linewidth=2 , label="True")
    plt.plot(y_pred.squeeze(), linestyle="--",  label="Predicted")
    plt.title(output_type)
    plt.legend()
    plt.show()


def plot_losses(epoch_loss, output_type):
    plt.plot(epoch_loss, linewidth=2)
    plt.title(output_type)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.show()


def data_split(X, Y):
    train_X, test_X, train_Y, test_Y = train_test_split(
        X, Y, test_size=0.2, random_state=0
    )

    test_X, val_X, test_Y, val_Y = train_test_split(
        test_X, test_Y, test_size=0.5, random_state=0
    )  # split test dataset into 2 halves, one for test and one for validation

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

# Write your function here


# ---
#
# ## Linear Regression using stochastic gradient descent via Numpy Arrays
#


class linear_regression_network(object):

    """
    We will implement Linear Regression Network as a python class. 
    Object of this class will have a few attributes for example learnable parameters
    and functions such as feed_forward(), l2_loss() etc.
    
    """

    # Initialize attributes of object
    def __init__(self, n_features=8):

        # No. of input features
        self.n_features = n_features

        # Learnable weights
        # initial random theta

        # Size of theta = dimension of x (+1 for bias)
        self.theta = np.random.rand(n_features+1)

    # This function just prints a few properties of object created from this class
    def __str__(self):

        msg = "Linear Regression:\n\nSize of Input = " + str(self.n_features)

        return msg

    # Read section#5.4.1 for the help
    def feed_forward(self, X):
        # Write linear function here
        # using the equation y = mx+b for linear regression and taking b(bias) to be zero
        y_hat = np.sum(X*self.theta, axis=1, keepdims=True)
        
        return y_hat   

    # Read section#5.4.2 for the help
    def l2_loss(self, Y, y_hat):
        loss = np.sum(np.power(y_hat - Y, 2)/len(Y))

        return loss

    # Read section#5.4.3 for the help
    def compute_gradient(self, y_hat, X, Y):

        # Batch size is the number of training examples we are considering in one epoch
        batch_size = y_hat.shape[0]
     
        # Record gradients for all examples in given batch        

        # By chain rule, gradient on weights is the product of local gradient on weights times
        # the incoming gradient, i.e., dloss/dw = dloss/dy_hat * dy_hat/dw
        # dy_hat/dw = x andd dloss/dy_hat = -(2/n) * Σ[(y(i) - y_hat(i)) * x(i)]
        # all of this calculation is summarized in one single vectorized step below:
        grad = -(2/batch_size) * (Y-y_hat).T @ X

        return grad

    def optimization(self, lr, grad):
        # Update theta using gradients
        self.theta = self.theta - lr * grad


def test_function(train_model, test_X):
        test_preds = train_model.feed_forward(test_X)
        return test_preds


# Write your code here


def train(model, trainX, trainY, valX, valY, n_epochs, lr, batch_size):
    # Write your training loop here
    # return model, epoch_loss, val_loss
    print(model)
    epoch_loss = []
    val_loss= []
    print("\n\nTraining...")
    epoch_loss = []
    val_loss = []
    for epoch in range(n_epochs):
        loss=0

        # Your implementation
        # create list of random indices that will create our mini-batches
        idx = [i for i in range(len(trainX))]
        random.shuffle(idx)
        iter_data_loader = int(len(trainX)/batch_size)

        step = 0
        for i in range(iter_data_loader): # loop over all the batches in one epoch
            try:
                slice = idx[step:batch_size+step]
                step = step + batch_size
            except IndexError:  #all indices in one epoch have been traversed
                continue

            y_hat = model.feed_forward(trainX[slice])
            loss = model.l2_loss(trainY[slice], y_hat)
            epoch_loss.append(loss)
            grad = model.compute_gradient(y_hat , trainX[slice] , trainY[slice])
            model.optimization(lr, grad)

        val_preds = model.feed_forward(valX)
        loss = model.l2_loss(valY, val_preds)
        val_loss.append(loss)

    plot_predictions(valY, val_preds,"validation_pred")   
    print("\nDone.")    

    return model, epoch_loss, val_loss


def main():
    # Data loader
    dataset = fetch_california_housing()
    X = dataset.data
    Y = dataset.target[:, np.newaxis]

    Me = set_mean(X)

    St = set_standardDeviation(X)

    # Normalize
    X = normalize(X, Me, St)

    # Add bias column
    X = add_bias_column(X)

    # Split the dataset into training and testing and validation
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data_split(X, Y)

    net = linear_regression_network(n_features=8)

    # ## Train Network using stochastic gradient descent
    lr = 0.01
    n_epochs = 200
    batch_size = 10000
    model, epoch_loss, val_loss = train(
        net, train_X, train_Y, val_X, val_Y, n_epochs, lr, batch_size
    )
    plot_losses(epoch_loss,"train_loss")
    plot_losses(val_loss,"val_loss")

    # Save the model
    pickle.dump(model, open("model.pkl", "wb"))

    # Load the model
    train_model = pickle.load(open('model.pkl','rb'))

    # ## Test prediction Prediction
    y_pred = test_function(train_model, test_X)
    plot_predictions(test_Y, y_pred, output_type="test_pred")


if __name__ == "__main__":
    main()
