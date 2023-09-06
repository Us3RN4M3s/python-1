#!/usr/bin/env python
# coding: utf-8

# # https://youtu.be/qjmfQf4ntRU

# In[1]:


import tensorflow as tf


# # Define the function 

# In[2]:


def f(x, y, z):
    return x*y + z**3 + x


# # Define and initialize x, y, and z
# 

# In[3]:


x = tf.Variable(-1.0)
y = tf.Variable(2.0)
z = tf.Variable(3.0)


# In[4]:


with tf.GradientTape() as tape: #using gradient tape output variable will hold output of the function
    output = f(x, y, z)


# In[5]:


derf_dx, derf_dy, derf_dz = tape.gradient(output, [x, y, z]) 
#output will be the result of the gradient tape as it records and gets the derivative of x,y,z
# stores x, y, and z in derf_dx, derf_dy, and derf_dz respectively 


# In[6]:


# Print the results as numpy objects 
print("the following partial derivates are x, y, and z respectively = ", derf_dx.numpy(), derf_dy.numpy(), derf_dz.numpy())


# ![Capture12.JPG](attachment:Capture12.JPG)

# following code is from Deep learning with python (second edition) by francois collet 

# ## Dense class

# In[7]:


import tensorflow as tf # here we have a dense class that takes a weight, bias, and activation function
import numpy as np
class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
                                                    #we get the shape of the weight
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1) #initalize weights randomly 
        self.W = tf.Variable(w_initial_value)
                                                    #we get the shape of the bias 
        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)
                                #using matrix multiplication of weights by the input and adding the bias 
    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b] # return the weights and bias 


# ### sequential class

# In[8]:


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
           x = layer(x)
        return x
# sequential class that holds the layers and within in each layer, 
#the weights get updated by adding them to the layer weights 
    @property
    def weights(self):
       weights = []
       for layer in self.layers:
           weights += layer.weights
       return weights


# In[9]:


model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4 #here we create the model with input size and neurons and layers, 
#check if the length of the weights is 4 


# ### batch generator

# In[10]:


import math
#creating the batch generator, assert that length of images = labels 
class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size) 
        #number of batches is the ceiling of the images/ batch size 
        #so the number of batches and batch size cant be greater than the number of images 

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels
    #goes through the next batch using index and next batch size and iterates through the batches 


# ### one training step

# In[11]:


def one_training_step(model, images_batch, labels_batch): 
    with tf.GradientTape() as tape: #the forward pass, computes predictions under gradienttape scope 
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions) 
        #per sample loss  = getting the loss by comparing the predictions with labels 
        average_loss = tf.reduce_mean(per_sample_losses)
        #average loss will be the mean of the total number of per sample losses 
    gradients = tape.gradient(average_loss, model.weights) 
    
    #gradient function is differentiable, using calculus chain rule of derivation to find thegradient function
    #mapping the current parameters and current batch of data to a gradient value BACKPROPAGATION
    
    #compute gradient of loss with respect to weights
    update_weights(gradients, model.weights)
    #the weights get updating by going opposite of the gradient and using the loss to update the weights to 
    #have a lower loss (will be defined in next cell) 
    return average_loss


# In[12]:


learning_rate = 1
# the rate that it learns (usually really small)
def update_weights(gradients, weights): #defining the function that updates the weights 
    learning_rate = learning_rate /2
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate) 
#assign sub will get the gradient and subtract so it is opposite 
#of the gradient by subtracting by the product of the learning rate and the zip (gradients,weights)


# In[13]:


from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3) 
#using the stochastic gradient descent with learning rate 
def update_weights(gradients, weights): 
#optimizer specifies the way the gradient of the loss will be used tp update parameters 
    optimizer.apply_gradients(zip(gradients, weights)) #apply the optimizer to the update_weights 


# ## full training loop

# In[14]:


def fit(model, images, labels, epochs, batch_size=128): 
#defining fit function to loop through n number of epochs and n number of batches
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
#loops through the batches and epochs 
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}") #prints the loss per batch 


# In[15]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)


# ## evaluating the model

# In[16]:


predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")


# In[ ]:




