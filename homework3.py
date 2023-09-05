#!/usr/bin/env python
# coding: utf-8

# # https://youtu.be/HDUi-jIGjaQ

# # Load Data

# In[1]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
from tensorflow import keras
from tensorflow.keras import layers


# # Run at 512 neurons

# In[2]:


x = 512 #verify that it works correctly as is 
model = keras.Sequential([
layers.Dense(x, activation="relu"), # relu activation 
layers.Dense(10, activation="softmax") # softmax acivation to send results through 10 neurons (for classifying)
])

model.compile(optimizer="rmsprop", #root mean square error propagation 
loss="sparse_categorical_crossentropy", # loss rate 
metrics=["accuracy"]) #measure accuracy of training on test data and measuring accuracy 

train_images = train_images.reshape((60000, 28 * 28)) # reduce dimensions to match sequential 
train_images = train_images.astype("float32") / 255 # dividing by 255 to match the grayscale data (0-255)
test_images = test_images.reshape((10000, 28 * 28)) # reshape like training images to match, both training  and testing must undergo same effects 
test_images = test_images.astype("float32") / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128) # fitting the model to go through each epoch or loop at a batch size of 128 samples 

test_loss, test_acc = model.evaluate(train_images, train_labels) #evaluating training 
print(f"test_acc: {test_acc}")

test_loss, test_acc = model.evaluate(test_images, test_labels) #evaluating testing 
print(f"test_acc: {test_acc}")


# # Function

# In[4]:


def Model(x,y,z):
    model = keras.Sequential([
    layers.Dense(x, activation="relu"),
    layers.Dense(y, activation="relu"),
    layers.Dense(z, activation="relu"),
    layers.Dense(10, activation="softmax")
])
        
        
    modelc = model.compile(optimizer="rmsprop",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])
    
    
    
    newmodel = model.fit(train_images, train_labels, epochs=5, batch_size=128)
    train_loss, test_acc = model.evaluate(train_images, train_labels) #evaluating training 
    print(f"train_acc: {test_acc}")

    test_loss, test_acc = model.evaluate(test_images, test_labels) #evaluating testing 
    print(f"test_acc: {test_acc}")
    return newmodel, modelc




# In[5]:


Model(3,10,50)


# In[6]:


def Model2(x, y): #give it the number of neurons and the number of hidden layers
    model = keras.Sequential()
    y = 1
    for i in range(y):
        model.add(keras.layers.Dense(x,activation = "relu"))
    model.add(keras.layers.Dense(10,activation = "softmax"))
        
    modelc = model.compile(optimizer="rmsprop",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])
    
    
    newmodel = model.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = model.evaluate(train_images, train_labels) #evaluating training 
    print(f"test_acc: {test_acc}")

    test_loss, test_acc = model.evaluate(test_images, test_labels) #evaluating testing 
    print(f"test_acc: {test_acc}")
    return newmodel, modelc



# In[7]:


Model2(640,200)


# In[ ]:





# In[2]:


train_images = train_images.reshape((60000, 28 * 28)) # reduce dimensions to match sequential 
train_images = train_images.astype("float32") / 255 # dividing by 255 to match the grayscale data (0-255)
test_images = test_images.reshape((10000, 28 * 28)) # reshape like training images to match, both training  and testing must undergo same effects 
test_images = test_images.astype("float32") / 255


# # Looping 10 times

# In[3]:


for x in range(1,11,1): #using a simple for loop to iterate through the number of neurons used from 1 - 10
    print(f"{x} neurons\n")
    model = keras.Sequential([
    layers.Dense(x, activation="relu"),
    layers.Dense(x, activation="relu"),
    layers.Dense(x, activation="relu"),
    layers.Dense(x, activation="relu"),
    layers.Dense(x, activation="relu"),
        
    layers.Dense(10, activation="softmax")])
    
  

    
    model.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

  

    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    train_loss, train_acc = model.evaluate(train_images, train_labels)
    print(f"train_acc: {train_acc}\n")

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"test_acc: {test_acc}\n")


# In[ ]:





# In[8]:





# In[ ]:




