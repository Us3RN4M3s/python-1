#!/usr/bin/env python
# coding: utf-8

# 

# ## original model

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# In[2]:


model.summary()


# In[3]:


with tf.device('/Cpu:0'):
    from tensorflow.keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255
    model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model1 = model.fit(train_images, train_labels, epochs=5, batch_size=64)


# In[4]:


with tf.device('/Cpu:0'):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.3f}")


# In[ ]:





# 

# In[ ]:





# ## strides no max pooling

# In[6]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32,strides=(2,2), kernel_size=3, activation="relu")(inputs)

x = layers.Conv2D(filters=64,strides=(2,2), kernel_size=3, activation="relu")(x)

x = layers.Conv2D(filters=128,strides=(2,2), kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# In[7]:


with tf.device('/Cpu:0'):
    from tensorflow.keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255
    model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model2 =  model.fit(train_images, train_labels, epochs=5, batch_size=64)


# In[8]:


with tf.device('/Cpu:0'):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.3f}")


# In[ ]:





# ### conclusions

# strides can help reduce spatial resolution of the feature maps, with a stride of 2,2, strides can also help with translation variance without the need for pooling because we are downsampling the feature maps. we reduce the sensitivity of the network to small shifts in the input, we can also retain more information about the input in the feature maps 

# In[ ]:





# In[ ]:





# ## average pooling 

# In[ ]:





# In[10]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.AveragePooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.AveragePooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# In[11]:


with tf.device('/Cpu:0'):
    from tensorflow.keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255
    model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model3 = model.fit(train_images, train_labels, epochs=5, batch_size=64)


# In[12]:


with tf.device('/Cpu:0'):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.3f}")


# In[ ]:





# ### conclusions

# maxpooling is more likely to perform better than average pooling because with this specific dataset, we arent too concerned with the specific position of the digit and more focused on the overall structure of the digit. using the max pooling we retain the structure and it will reduce the sensitivity to small translations while average pooling will get the average of the pool and dilute the structure/ features of the digit. max pooling will preserve the spatial features of the input better when compared to the average pooling while may smooth out features too much and make it more difficult to identify sublte differences between digits, an example of this is the textbook example of a 6 digit's features being smoothed so much it becomes an 8

# In[ ]:





# ## replace maxpooling w/ conv2D & strides=3, all filters same number and strides alternate between 1 and 3

# In[14]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32,strides=1, kernel_size=3, activation="relu")(inputs)
x = layers.Conv2D(filters=32,strides=3, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=32,strides=1, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=32,strides=3, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=32,strides=1, kernel_size=3, activation="relu", padding='same')(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# In[15]:


with tf.device('/Cpu:0'):
    from tensorflow.keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255
    model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model4=   model.fit(train_images, train_labels, epochs=5, batch_size=64)


# In[16]:


with tf.device('/Cpu:0'):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.3f}")


# In[ ]:





# ### conclusion

# alternating between strides of 1 and 3 could cayse a loss of spatial information in feature maps which make it harder to distinguish features of the digits. using the same number of filters across all layers may not be enough to gather all the features in deeper layers as some features may require different layers, it will be limited to just that number of filters. Maxpooling prevents the network from being sensitive to small translations in the input, maxpooling selects the max value in each pool region which can contain important features. 
