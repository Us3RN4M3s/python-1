#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

maxlen = 200
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

vocab_size = max_words + 1  # 1 is added for the padding token


# In[3]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, 
                                                  stratify=y_train, train_size = 5000)


# In[4]:


inputs = Input(shape=(maxlen,))
x = Embedding(vocab_size, 128, input_length=maxlen)(inputs)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)


outputs = Dense(1, activation='sigmoid')(x)

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[5]:


batch_size = 512
epochs = 100

model1 = model.fit(x_train, y_train, batch_size=batch_size, 
                   epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks_list )


# In[6]:


print('evaluate on training data')
results = model.evaluate(x_train, y_train, batch_size = 128)
print(' loss, acc:', results)
print('evaluate on validation data')
results = model.evaluate(x_val, y_val, batch_size = 128)
print(' loss, acc:', results)


# In[7]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[8]:


import matplotlib.pyplot as plt
train_acc = model1.history["accuracy"]
loss_values = model1.history["loss"]
val_loss = model1.history["val_loss"]
val_acc = model1.history["val_accuracy"]
epochs = range(1, 30)

plt.plot(epochs, loss_values, "b", label="training loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[10]:


max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

maxlen = 200
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

vocab_size = max_words + 1  # 1 is added for the padding token


# In[11]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, 
                                                  stratify=y_train, train_size = 10000)


# In[12]:


inputs = Input(shape=(maxlen,))
x = Embedding(vocab_size, 128, input_length=maxlen)(inputs)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)


outputs = Dense(1, activation='sigmoid')(x)

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[13]:


batch_size = 512
epochs = 100

model2 = model.fit(x_train, y_train, batch_size=batch_size, 
                   epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks_list)


# In[15]:


import matplotlib.pyplot as plt
val_loss = model2.history["val_loss"]
loss_values = model2.history["loss"]
val_acc = model1.history["val_accuracy"]
epochs = range(1, 14)
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[14]:


print('evaluate on training data')
results = model.evaluate(x_train, y_train, batch_size = 128)
print(' loss, acc:', results)
print('evaluate on validation data')
results = model.evaluate(x_val, y_val, batch_size = 128)
print(' loss, acc:', results)


# In[15]:


max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

maxlen = 200
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

vocab_size = max_words + 1  # 1 is added for the padding token


# In[16]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, 
                                                  stratify=y_train, train_size = 15000)


# In[17]:


inputs = Input(shape=(maxlen,))
x = Embedding(vocab_size, 128, input_length=maxlen)(inputs)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)


outputs = Dense(1, activation='sigmoid')(x)

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[18]:


batch_size = 512
epochs = 100

model3 = model.fit(x_train, y_train, batch_size=batch_size,
                   epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks_list)


# In[19]:


import matplotlib.pyplot as plt
val_loss = model3.history["val_loss"]
loss_values = model3.history["loss"]
val_acc = model1.history["val_accuracy"]
epochs = range(1, 9)
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")

plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[20]:


print('evaluate on training data')
results = model.evaluate(x_train, y_train, batch_size = 128)
print(' loss, acc:', results)
print('evaluate on validation data')
results = model.evaluate(x_val, y_val, batch_size = 128)
print(' loss, acc:', results)


# In[21]:


max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

maxlen = 200
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

vocab_size = max_words + 1  # 1 is added for the padding token


# In[22]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, 
                                                  train_size = 20000)


# In[23]:


inputs = Input(shape=(maxlen,))
x = Embedding(vocab_size, 128, input_length=maxlen)(inputs)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)


outputs = Dense(1, activation='sigmoid')(x)

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[24]:


batch_size = 512
epochs = 100

model4 = model.fit(x_train, y_train, batch_size=batch_size,
                   epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks_list)


# In[29]:


import matplotlib.pyplot as plt
val_loss = model4.history["val_loss"]
loss_values = model4.history["loss"]
val_acc = model1.history["val_accuracy"]
epochs = range(1, 17)
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")

plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[25]:


print('evaluate on training data')
results = model.evaluate(x_train, y_train, batch_size = 128)
print(' loss, acc:', results)
print('evaluate on validation data')
results = model.evaluate(x_val, y_val, batch_size = 128)
print(' loss, acc:', results)


# In[26]:


max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

maxlen = 200
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

vocab_size = max_words + 1  # 1 is added for the padding token


# In[27]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train  
                                                  )


# In[28]:


inputs = Input(shape=(maxlen,))
x = Embedding(vocab_size, 128, input_length=maxlen)(inputs)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)


outputs = Dense(1, activation='sigmoid')(x)

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[29]:


batch_size = 512
epochs = 100

model5 = model.fit(x_train, y_train, batch_size=batch_size,
                   epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks_list)


# In[30]:


import matplotlib.pyplot as plt
val_loss = model5.history["val_loss"]
loss_values = model5.history["loss"]
val_acc = model1.history["val_accuracy"]
epochs = range(1, 9)
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")

plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[31]:


print('evaluate on training data')
results = model.evaluate(x_train, y_train, batch_size = 128)
print(' loss, acc:', results)
print('evaluate on validation data')
results = model.evaluate(x_val, y_val, batch_size = 128)
print(' loss, acc:', results)


# In[ ]:





# In[32]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
loss_values = model1.history["loss"]
loss_values2 = model2.history["loss"]
loss_values3 = model3.history["loss"]
loss_value4 = model4.history["loss"]
loss_value5 = model5.history["loss"]

val_loss = model1.history["val_loss"]
val_loss2 = model2.history["val_loss"]
val_loss3 = model3.history["val_loss"]
val_loss4 = model4.history["val_loss"]
val_loss5 = model5.history["val_loss"]

#train_acc = model1.history["accuracy"]
#val_acc = model1.history["val_accuracy"]
epochs = range(1, 30)
epochs2 = range(1,14)
epochs3 = range(1,9)
epochs4 = range(1,17)
epochs5 = range(1,9)

plt.plot(epochs, loss_values, "b", label="training loss 5000")
plt.plot(epochs2, loss_values2, "m", label="training loss 10000")
plt.plot(epochs3, loss_values3, "r", label="training loss 15000")
plt.plot(epochs4, loss_value4, "k", label="training loss 20000")
plt.plot(epochs5, loss_value5, "c", label="training loss 25000")

plt.plot(epochs, val_loss, "b", label="validation loss 5000")
plt.plot(epochs2, val_loss2, "m", label="validation loss 10000")
plt.plot(epochs3, val_loss3, "r", label="validation loss 15000")
plt.plot(epochs4, val_loss4, "k", label="validation loss 20000")
plt.plot(epochs5, val_loss5, "c", label="training loss 25000")




plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


# In[ ]:




