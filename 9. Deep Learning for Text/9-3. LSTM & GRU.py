# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="A6qBaakgHY0r" outputId="8d414eb4-ce2a-460f-ef22-bfa5d46bfe49"
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words = 500)
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size = 0.2, random_state = 42)

# %% id="wg0OsqBYIIHf"
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen = 100)
val_seq = pad_sequences(val_input, maxlen = 100)
test_seq = pad_sequences(test_input, maxlen = 100)

# %% id="-l0E2oFeKvCC"
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length = 100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# %% colab={"base_uri": "https://localhost:8080/"} id="15NhNFz-LFLl" outputId="76d71f84-752a-4ebb-d4d8-23c64f7ce1c6"
model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="V1C7K1hLLF6P" outputId="89685d09-2461-4e05-9ad3-85efe763e4cb"
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
earlystopping_cb = keras.callbacks.EarlyStopping(patience= 3, restore_best_weights= True)
history = model.fit(train_seq, train_target, epochs = 100, batch_size = 64,
                    validation_data = (val_seq, val_target),
                    callbacks = [checkpoint_cb, earlystopping_cb])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="XcK_RXkHMMiR" outputId="c18524ac-31f3-4712-fba9-1d0838a916ca"
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# %% id="1H7dGhs8M1nY"
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length = 100))
model2.add(keras.layers.LSTM(8, dropout = 0.3))
model2.add(keras.layers.Dense(1, activation = 'sigmoid'))

# %% colab={"base_uri": "https://localhost:8080/"} id="eR-T967-NRGV" outputId="21d9accc-1186-44d1-d6ed-3ed2d2312fba"
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model2.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-w-dropout-model.h5')
earlystopping_cb = keras.callbacks.EarlyStopping(patience= 3, restore_best_weights= True)
history = model2.fit(train_seq, train_target, epochs = 100, batch_size = 64,
                    validation_data = (val_seq, val_target),
                    callbacks = [checkpoint_cb, earlystopping_cb])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="hDYfht40Ni6Q" outputId="02da5384-641a-4070-bfea-8b150cef0102"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# %% id="kWAbJaK3N9Xk"
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length = 100))
model3.add(keras.layers.LSTM(8, dropout = 0.3, return_sequences = True))
model3.add(keras.layers.LSTM(8, dropout = 0.3))
model3.add(keras.layers.Dense(1, activation = 'sigmoid'))

# %% colab={"base_uri": "https://localhost:8080/"} id="zw2VNlh3OZ9Y" outputId="eb62f241-4e1b-44f6-c4a8-89e1a0dba3de"
model3.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="KSX-bMZJObqV" outputId="01f1d723-eea4-4761-f682-2294dc32e723"
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model3.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2LSTM-model.h5')
earlystopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)
history = model3.fit(train_seq, train_target, epochs = 100, batch_size = 64,
                     validation_data = (val_seq, val_target),
                     callbacks = [checkpoint_cb, earlystopping_cb])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="53urbE9dPhBP" outputId="c6b7430e-a761-482f-94bf-6069626373bd"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# %% id="LX7_GkzLPpg6"
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length = 100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation = 'sigmoid'))

# %% colab={"base_uri": "https://localhost:8080/"} id="ZHB-eaT1P23r" outputId="bee35f30-0d62-4215-c3ea-7b536fb6f481"
model4.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="5hwttNGsRAZo" outputId="360c1342-0e46-4e2b-d4ae-b718c868e83b"
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model4.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.h5')
earlystopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)
history = model4.fit(train_seq, train_target, epochs = 100, batch_size = 64,
                     validation_data = (val_seq, val_target),
                     callbacks = [checkpoint_cb, earlystopping_cb])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="HMoTycPnRMiM" outputId="65bb7785-e075-40ff-f2e7-83fd0fb99338"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="54rm6OGnRNsN" outputId="8c6c44a2-68ef-45d6-8fb4-622afd9da469"
rnn_model = keras.models.load_model('best-2LSTM-model.h5')
rnn_model.evaluate(test_seq, test_target)
