import tensorflow as tf
from src.titanic import data_loader
import numpy as np
import src.titanic.graph_plotter as gp
import os

# Train data
data_path='train.csv'
train_in,train_out,test_in,test_out = data_loader.loadNclean_dataForTf(datapath=data_path)

# For kaggle competition
test_path = 'test.csv'
submission_test_in = data_loader.loadNclean_dataForTf_Kaggle(testpath=test_path)

# Configure input nodes for input layer
input_layer_nodes=len(train_in[0])

# Configure output nodes for output layer
output_layer_nodes=len(train_out[0])

# Set number of hidden layer nodes for each hidden layer
h1 = 30
h2 = 20
h3 = 15
h4 = 10

learning_rate = 0.01
epoch =5000

# l2 regularization parameter
lmbda = 0.01

# Boolean flag for session restore
restore=False

# Set path to save session
save_path = os.path.dirname(os.path.abspath(__file__))+'/saved_Sessions/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_path = save_path+'titanic_tf_model.sess'

# Input/output layers
input_layer=tf.placeholder(shape=[None,input_layer_nodes],dtype=tf.float32)
output_layer=tf.placeholder(shape=[None,output_layer_nodes],dtype=tf.float32)

# Set Seed value for generations of weights and biases
seed = 1

# weights/biases
weights = {
    'w1':tf.Variable(tf.random_normal([input_layer_nodes,h1],dtype=tf.float32)),
    'w2':tf.Variable(tf.random_normal([h1,h2],dtype=tf.float32)),
    'w3':tf.Variable(tf.random_uniform([h2,h3],dtype=tf.float32,seed=seed)),
    'w4':tf.Variable(tf.random_uniform([h3,h4],dtype=tf.float32,seed=seed)),
    'w5':tf.Variable(tf.random_uniform([h4,output_layer_nodes],dtype=tf.float32,seed=seed))
}
biases= {
    'b1':tf.Variable(tf.zeros([h1],dtype=tf.float32)),
    'b2':tf.Variable(tf.zeros([h2],dtype=tf.float32)),
    'b3':tf.Variable(tf.zeros([h3],dtype=tf.float32)),
    'b4':tf.Variable(tf.zeros([h4],dtype=tf.float32)),
    'b5':tf.Variable(tf.zeros([output_layer_nodes],dtype=tf.float32))
}

# Model
def model(x,weights,biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['w1']),biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['w2']), biases['b2']))
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['w3']), biases['b3']))
    layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['w4']), biases['b4']))
    return tf.nn.sigmoid(tf.add(tf.matmul(layer4,weights['w5']),biases['b5']))

# Predictions/Activations
pred = model(input_layer, weights, biases)

# l2 regularization
l2 = ((tf.nn.l2_loss(weights['w1']))+(tf.nn.l2_loss(weights['w2']))+(tf.nn.l2_loss(weights['w3']))+(tf.nn.l2_loss(weights['w4'])))*lmbda

# Quadratic cost
cost = (tf.reduce_mean(tf.square(output_layer - pred)/2))+(l2*lmbda)

# Cross entropy cost
# cost = tf.reduce_mean(((output_layer * tf.log(pred))+(1-output_layer*tf.log(1-pred)))* -1)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    # Create a session saving/restore variable
    saver = tf.train.Saver()

    cost_list=[]
    accuracy_list=[]

    # Initialize all variables
    sess.run(tf.initialize_all_variables())

    # If restore=True , then restore the model data , run predictions on test_in,test_out
    if restore:
        print ('Restoring model data')
        model_restore = saver.restore(sess=sess,save_path=save_path)

        # Running the restored model on test data for predictions
        c, p = (sess.run([cost, pred], feed_dict={input_layer: test_in, output_layer: test_out}))
        accuracy = [(np.argmax(x), np.argmax(y)) for x, y in zip(p, test_out)]
        results = (sum(int(x == y) for x, y in accuracy) / len(test_in)) * 100
        print ("Accuracy on {0} test samples: {1}%".format(len(test_in),results))

    else:
        # If restore=False , then train the model for range of epochs
        for i in range(epoch):
            # Feed the model with 'train_in' and calculate the cost , then optimize it (Basically train the model)
            c,_,p=sess.run([cost,optimizer,pred],feed_dict={input_layer:train_in,output_layer:train_out})
            cost_list.append(c)
            accuracy = [(np.argmax(x), np.argmax(y)) for x, y in zip(p, train_out)]
            results = (sum(int(x == y) for x, y in accuracy) / len(train_in)) * 100
            accuracy_list.append(results)
            print("Epoch:{0} --> {1}%  ".format(i, results))

        # Running the trained model on test data for predictions
        c, p = (sess.run([cost, pred], feed_dict={input_layer: test_in, output_layer: test_out}))
        accuracy = [(np.argmax(x), np.argmax(y)) for x, y in zip(p, test_out)]
        results = (sum(int(x == y) for x, y in accuracy) / len(test_in)) * 100
        print("Accuracy on {0} test samples: {1}%".format(len(test_in), results))

        # Save the trained model in 'save_path'
        save_path = saver.save(sess=sess,save_path=save_path)

        # Display cost vs epoch graph
        gp.cost_vs_epoch(cost_list)

        # Display accuracy vs epoch graph
        gp.accuracy_vs_epoch(accuracy_list)

        # For kaggle
        p = sess.run(pred, feed_dict={input_layer: submission_test_in})
        submission_predictions = []
        for x in p:
            submission_predictions.append(np.argmax(x))
        data_loader.prep_submission_file(test_path, submission_predictions)
