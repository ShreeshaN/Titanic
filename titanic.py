from src.titanic import data_loader
import numpy as np
from src.titanic import graph_plotter


# Hyper parameters of the model
learning_rate = 0.01
epochs =500
network_layers = [6,10,5,2]
lmbda = 0.1
l2_regularization = True

# 1. Cost Function - Quadratic cost
# 2. Optimizer - Gradient Descent
# 3. Activation Function - Sigmoid
# 4. Regularization - l2

class Network:
    def __init__(self,sizes):
        np.random.seed(1)
        self.num_of_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost_per_sample_list=[]
        self.accuracy_list=[]
        self.final_cost_list=[]

    """ Calculates the activation of a neuron using sigmoid function """
    def sigmoid(self,z):
        z = z.astype(float)
        return 1.0 / (1.0 + np.exp(-(z)))

    """ Calculates the derivative of the sigmoid function """
    def sigmoid_prime(self,z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    """  Partial derivative of cost function with respect to activation will give this below equation) """
    def cost_derivative(self, activation, y):
        return (activation - y)

    """ Quadratic cost equation .
    Instead of splitting this function into two , one to calculate just cost for each training sample
    and the other to calculate cost for each epoch ,its been combined and written for convinience"""
    def quadratic_cost_func(self,activation,y,cost_list=None,final=False):
        if not final:
            return (activation-y)**2
        else:
            return (0.5*cost_list)

    """ Take each training sample , feedforward , calculate error using cost function , backprop to find
    new set of weights and biases to be updated in the network"""
    def backprop(self,x,y):
        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]
        activation =x
        activations=[x]
        zs=[]

        # Calculate weighted output,store it in 'zs' and calculate activation and store it in 'activations'
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Calculate cost_per_sample
        cost_per_sample = self.quadratic_cost_func(activation,y)

        # Calculate the error rate using the quadratic cost function value
        delta = self.cost_derivative(activation,y) * self.sigmoid_prime(zs[-1])

        # Update weights and biases of the final layer
        new_biases[-1] = delta
        new_weights[-1] = np.dot(delta,activations[-2].transpose())

        for l in range(2, self.num_of_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(),delta) * self.sigmoid_prime(z)
            new_biases[-l] = delta
            new_weights[-l] = np.dot(delta,activations[-l-1].transpose())


        return (new_biases, new_weights,cost_per_sample)


    """ Update weights and biases with learning rate and regularization parameter """
    def update_biases_weights_l2_reguralization(self, new_biases, new_weights):
        self.weights = [(1-(learning_rate*lmbda)/len(train_data))*(w - (learning_rate * del_w)) for w, del_w in zip(self.weights, new_weights)]
        self.biases = [b - (learning_rate * del_b) for b, del_b in zip(self.biases, new_biases)]

    """ Update weights and biases with a learning rate
        If l2_regularization is true then apply l2_reg and update
    """
    def update_biases_weights(self,new_biases,new_weights,l2_regularization=False):
        if(l2_regularization):
            self.weights = [(1 - (learning_rate * lmbda) / len(train_data)) * (w - (learning_rate * del_w)) for w, del_w
                            in zip(self.weights, new_weights)]
            self.biases = [b - (learning_rate * del_b) for b, del_b in zip(self.biases, new_biases)]
        else:
            self.weights = [w - (learning_rate*del_w) for w, del_w in zip(self.weights, new_weights)]
            self.biases = [b - (learning_rate*del_b) for b, del_b in zip(self.biases, new_biases)]

    """ Given input sample calculate activations of all layers"""
    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    """ Evaluate model performance based with test_data """
    def evaluate(self,test_data):
        test_results_pred=[]
        test_results_actual=[]
        for x in test_data:
            s = np.argmax(self.feedforward(x[0]))
            test_results_pred.append(s)
            test_results_actual.append(np.argmax(x[1]))
        return sum(int(x == y) for (x, y) in zip(test_results_pred,test_results_actual))

if(__name__=='__main__'):
    train_filepath = 'train.csv'
    train_data ,test_data = data_loader.loadNclean_data(train_filepath)

    net = Network(network_layers)
    test_len = len(test_data)
    for x in range(epochs):
        for data in train_data:

            new_biases, new_weights, cost_per_sample = net.backprop(data[0], data[1])
            net.update_biases_weights(new_biases, new_weights,l2_regularization)

            # Each cost sample has 2 values coming from 2 output neurons , so finding the mean of them
            net.cost_per_sample_list.append(sum(cost_per_sample) / len(cost_per_sample))

        # Calculate the quadratic cost for each epoch
        cost_per_epoch = net.quadratic_cost_func(None, None,
                                                 (sum(net.cost_per_sample_list) / float(len(net.cost_per_sample_list))),
                                                 True)
        net.final_cost_list.append(cost_per_epoch)
        # Empty the list
        net.cost_per_sample_list = []

        # Calculate the accuracy by passing the test_samples , print each epoch accuracy to observe how it varies
        evaluated = net.evaluate(test_data)
        accuracy = (float(evaluated) / test_len) * 100
        net.accuracy_list.append(accuracy)
        print("Epoch {0} : {1}/{2}->{3}".format(x, evaluated, test_len, accuracy))

    # Cost vs Epoch graph - train's cost
    graph_plotter.cost_vs_epoch(net.final_cost_list)

    # Accuracy vs Epoch graph - test's accuracy
    graph_plotter.accuracy_vs_epoch(net.accuracy_list)
