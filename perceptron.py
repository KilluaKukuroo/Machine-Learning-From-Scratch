
import numpy as np


## activation function, e.g., sigmoid
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

## derivative of sigmoid function
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))



'''
- Training with SGD, suppose we one sample a time, shape of X = [num_features,],  e.g., [4,]
- X is a vector, not a matrix
- W is a matrix, shape = [num_classes, num_features], e.g., [1,4]
- b is a vector, shape = [num_output], e.g., [1,]

leason learned:
    - learning rate is very important, if your acc and loss don't change, increase the lr; 

'''
class Perceptron():

    def __init__(self, num_instances, num_features, num_classes):

        self.W = np.random.uniform(low=-0.01, high=0.01, size=(num_classes, num_features))
        # self.b = np.random.uniform(low=0, high=1, size=())
        self.b = np.zeros(num_classes)
        self.lr = 0.01

        self.dw = []

    ## loss function, e.g., MSE loss
    def loss(self, y_pred, y_true):
        return 1/2 * (y_pred - y_true) ** 2

    ## derivative of MSE loss function
    def dloss(self, y_pred, y_true):
        return y_pred - y_true

    def forward(self, X):
        # print(X.shape, self.W.shape, self.b.shape)
        out = np.dot(X, self.W.T) + self.b
        y_pred = sigmoid(out)
        return y_pred

    ## suppose we one sample a time, shape of X = [num_features, 1]
    def backpropagation(self, X, y_pred, y_true):
        err = self.loss(y_pred, y_true)
        dEdy_pred = self.dloss(y_pred, y_true)

        

        y_pred = self.forward(X)
        Delta = dEdy_pred * y_pred * (1 - y_pred)  ## dsigmoid(out)
        dEdW = Delta * X  ## shape = [num_classes, num_features] [1,4]
        dEdb = Delta  # np.sum(Delta, axis=0)
        # dEdX = np.dot(Delta, self.W)

        # print(err, dEdy_pred, Delta)

        # print("================")
        # print(y_pred.shape, Delta.shape, dEdW.shape, dEdb.shape)
        # print(dEdW)
        # print(Delta)
        # print(X)
        self.dw.append(dEdW)
        return dEdW, dEdb

    def train(self, X, y_true):
        y_pred = self.forward(X)
        dEdW, dEdb = self.backpropagation(X, y_pred, y_true)

        # print(dEdW)
        # print(dEdW.shape)

        ## parameter updata
        self.W = self.W - self.lr * dEdW
        self.b = self.b - self.lr * dEdb

    def accuracy(self, X, y_true):
        y_pred = self.forward(X)
        loss = np.mean(self.loss(y_pred, y_true))
        ## another way to calculate acc
        # num_correct = 0
        # for i in range(len(y_pred)):
        #     if(y_pred[i] > 0.5):
        #         y_pred[i] = 1
        #     else:
        #         y_pred[i] = 0
        #     if(y_true[i] == y_pred[i]):
        #         num_correct += 1
        # acc = num_correct / len(y_pred)

        y_pred = [1 if y_pred[i] > 0.5 else 0 for i in range(len(y_pred))]
        y_pred = np.array(y_pred).reshape(y_true.shape[0], y_true.shape[1]) # make tha shape of y_pred and y_true the same

        acc = np.mean(y_pred == y_true)

        return acc, loss



## hyperparameters
EPOCHS = 10
num_instances = 1
num_classes = 1
num_features = 4

## prepare the dataset
## data: if xi < 10, then y = 0; else y = 1;
x_train_1 = np.random.randint(1,4, (100, num_features))  ## 100 x (0<x<1) with num_features == 4
x_train_2 = np.random.randint(10,15, (100, num_features))
y_train_1 = np.zeros((100,1))
y_train_2 = np.ones((100,1))
x_train = np.vstack((x_train_1, x_train_2))
y_train = np.vstack((y_train_1, y_train_2))

x_test_1 = np.random.randint(3,9, size=(100, num_features))
x_test_2 = np.random.randint(14,20, size=(100, num_features))
x_test = np.vstack((x_test_1, x_test_2))
y_test_1 = np.zeros((100,1))
y_test_2 = np.ones((100, 1))
y_test = np.vstack((y_test_1, y_test_2))

# np.random.shuffle(x_train)  ## if shufle, X and Y have to be shuffled with the same seed to keep the label and input match


## create the Perceptron model
model = Perceptron(num_instances, num_features, num_classes)

## train the Perceptron
# print(x_train.shape, y_train.shape)
for i in range(EPOCHS):
    for j in range(len(x_train)):
        idx = np.random.randint(0,200)  # generate the index randomly to choose x[idx], y[idx]
        model.train(x_train[idx], y_train[idx])
    
    # calculate the acc, and loss
    acc, lo = model.accuracy(x_train, y_train)
    test_acc, test_lo = model.accuracy(x_test, y_test)
    
    print("In epoch #%d: train_loss: %f, train_acc: %f,, test_loss: %f, test_acc: %f" % (i, lo, acc, test_lo, test_acc))
    # print("dw: ", np.mean(model.dw))  ## monitor the gradient of w