#!/usr/bin/env python
# coding: utf-8



# In[1]:


from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# **Stratified Model**

# ## 5000 samples 

# In[2]:


from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test)  = imdb.load_data(num_words=10000)
# Split the data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_train, X_test), axis=0),
np.concatenate((y_train, y_test), axis=0),test_size=0.2,train_size = 5000, 
stratify=np.concatenate((y_train, y_test), axis=0))
#concatenating the rows X_train,X_test,y_train,y_test, 
#and stratify the concatinated y_train, Y_test (labels) 
# Split the training data into training and validation sets with stratified sampling
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)


# In[89]:





# In[4]:


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(X_train)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original = model.fit(train_data,y_train,
                             epochs=20, batch_size=512, validation_split=0.4)


# In[5]:


import matplotlib.pyplot as plt
val_loss = history_original.history["val_loss"]
loss_values = history_original.history["loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# ## 10000 samples

# In[98]:


from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test)  = imdb.load_data(num_words=10000)
# Split the data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_train, X_test), 
axis=0),np.concatenate((y_train, y_test), axis=0),test_size=0.2,train_size=10000, 
stratify=np.concatenate((y_train, y_test), axis=0))


# In[99]:


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(X_train)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original2 = model.fit(train_data, y_train,
                             epochs=20, batch_size=512, validation_split=0.4)


# In[100]:


import matplotlib.pyplot as plt
val_loss = history_original2.history["val_loss"]
loss_values = history_original2.history["loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# ## 15000 samples

# In[10]:


from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test)  = imdb.load_data(num_words=10000)
# Split the data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_train, X_test), 
axis=0),np.concatenate((y_train, y_test), axis=0),test_size=0.2,train_size=15000,
stratify=np.concatenate((y_train, y_test), axis=0))


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(X_train)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original3 = model.fit(train_data, y_train,
                             epochs=20, batch_size=512, validation_split=0.4)


# In[11]:


import matplotlib.pyplot as plt
val_loss = history_original3.history["val_loss"]
loss_values = history_original3.history["loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# ## 20000 samples

# In[86]:


from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test)  = imdb.load_data(num_words=10000)
# Split the data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_train, X_test), 
axis=0),np.concatenate((y_train, y_test), axis=0),test_size=0.2,train_size=20000, 
stratify=np.concatenate((y_train, y_test), axis=0))


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(X_train)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original4 = model.fit(train_data, y_train,
                             epochs=20, batch_size=512, validation_split=0.4)


# In[87]:


import matplotlib.pyplot as plt
val_loss = history_original4.history["val_loss"]
loss_values = history_original4.history["loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# ## 25000 samples

# In[14]:


from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test)  = imdb.load_data(num_words=10000)
# Split the data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_train, X_test), 
axis=0),np.concatenate((y_train, y_test), axis=0),test_size=0.2,train_size=25000, 
stratify=np.concatenate((y_train, y_test), axis=0))


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(X_train)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original5 = model.fit(train_data, y_train,
                             epochs=20, batch_size=512, validation_split=0.4)


# In[15]:


import matplotlib.pyplot as plt
val_loss = history_original5.history["val_loss"]
loss_values = history_original5.history["loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[ ]:





# In[ ]:





# In[88]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
loss_values1 = history_original.history["loss"]
loss_values2 = history_original2.history["loss"]
loss_values3 = history_original3.history["loss"]
loss_values4 = history_original4.history["loss"]
loss_values5 = history_original5.history["loss"]

val_loss1 = history_original.history["val_loss"]
val_loss2 = history_original2.history["val_loss"]
val_loss3 = history_original3.history["val_loss"]
val_loss4 = history_original4.history["val_loss"]
val_loss5 = history_original5.history["val_loss"]

epochs = range(1, 21)
plt.plot(epochs, val_loss1, "r", label="Validation loss 5000")
plt.plot(epochs, val_loss2, "g", label="Validation loss 10000")
plt.plot(epochs, val_loss3, "y", label="Validation loss 15000")
plt.plot(epochs, val_loss4, "k", label="Validation loss 20000")
plt.plot(epochs, val_loss5, "m", label="Validation loss 25000")


plt.plot(epochs, loss_values1, "r", label="Training loss 5000 samples")
plt.plot(epochs, loss_values2, "g", label="Training loss 10000 samples")
plt.plot(epochs, loss_values3, "y", label="Training loss 15000 samples")
plt.plot(epochs, loss_values4, "k", label="Training loss 20000 samples")
plt.plot(epochs, loss_values5, "m", label="Training loss 25000 samples")

plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()


# ## Smaller Model

# In[84]:


from tensorflow.keras.datasets import imdb
(train_data, train_labels), _ = imdb.load_data(num_words=5000)
def vectorize_sequences(sequences, dimension=5000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(train_data)

model = keras.Sequential([
    layers.Dense(5, activation="relu"),
    layers.Dense(5, activation="relu"),
    
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original7 = model.fit(train_data, train_labels,
                             epochs=20, batch_size=512, validation_split=0.4)


# In[85]:


import matplotlib.pyplot as plt
val_loss = history_original7.history["val_loss"]
val_loss2 = history_original.history["val_loss"]
loss_values = history_original7.history["loss"]
loss_values2 = history_original.history["loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss2, "k", label="Validation loss original")
plt.plot(epochs, loss_values2, "m", label="Training loss original")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# #### This technique mitigates the overfitting by reducing the models ability to memorize as quickly compared to a model with more neurons 

# ## Dropout

# In[70]:


from tensorflow.keras.datasets import imdb
(train_data, train_labels), _ = imdb.load_data(num_words=5000)
def vectorize_sequences(sequences, dimension=5000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(train_data)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.66),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.66),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original8 = model.fit(train_data, train_labels,
                             epochs=20, batch_size=512, validation_split=0.4)


# In[73]:


import matplotlib.pyplot as plt
val_loss = history_original8.history["val_loss"]
val_loss2 = history_original.history["val_loss"]
loss_values = history_original8.history["loss"]
loss_values2 = history_original.history["loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss2, "k", label="Validation loss original")
plt.plot(epochs, loss_values2, "m", label="Training loss original")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# #### Dropout proves effective as i gave it 0.66 which means each neuron has a 66% chance of beung retained and 34% chance of being dropped out and is used to prevent overfitting and improve generalization performance 

# In[ ]:





# In[ ]:




