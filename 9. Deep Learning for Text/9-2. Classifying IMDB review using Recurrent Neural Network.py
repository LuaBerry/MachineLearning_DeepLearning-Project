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

# %% id="81Nsfa2reQai"
from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words = 500)

# %% colab={"base_uri": "https://localhost:8080/"} id="voLMIzuU2r2L" outputId="856fd57f-8f87-4ea2-ca96-f0cfcf4194ff"
print(train_input.shape, test_input.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="tJm_kROa207g" outputId="eb6a9757-92c6-4f6f-ac0a-1b257b45b20c"
print(len(train_input[1]))

# %% colab={"base_uri": "https://localhost:8080/"} id="GbSIEwmo239S" outputId="b3bccae0-1031-4611-9f8b-146b797befa5"
print(train_input[0])

# %% colab={"base_uri": "https://localhost:8080/"} id="AoYtxBdE26or" outputId="f29627f7-ff28-410d-b823-cd372354225c"
print(train_target[:20])

# %% id="Si_AMhZqecme"
from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size = 0.2, random_state = 42)

# %% id="zmzkSdJZ3LKK"
import numpy as np
lengths = np.array([len(x) for x in train_input])

# %% colab={"base_uri": "https://localhost:8080/"} id="L_NcpowR3Q7T" outputId="b74e1fa4-f9cd-43e1-e5de-9a4722802776"
print(np.mean(lengths), np.median(lengths))

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="qT3N9Ol_3Vfr" outputId="bce5bbdc-109f-4c3f-d033-59ddc7daf568"
import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('freqency')
plt.show()

# %% id="bR4lByZEeuKl"
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen = 100)

# %% colab={"base_uri": "https://localhost:8080/"} id="rJS55wuu3uHR" outputId="b8ecc592-3202-4fe4-ba62-ca2054662f64"
print(train_seq.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="xR1isCQH3znc" outputId="4404e740-5980-408b-eeaf-7de458a84dfe"
print(train_seq[0])

# %% colab={"base_uri": "https://localhost:8080/"} id="SZk--gYm33br" outputId="ca77f6fb-10ab-40ce-a487-e0ad6175de7d"
print(train_input[0][-10:])

# %% colab={"base_uri": "https://localhost:8080/"} id="szHZpFDH36pC" outputId="ff43edb7-2b13-4788-c433-324b39079f6a"
print(train_input[0][:10])

# %% colab={"base_uri": "https://localhost:8080/"} id="2OpXIJ4M4IM4" outputId="f3c0b151-7485-478e-ee42-e85c2a369189"
print(train_seq[5])

# %% id="XVdQZBaSe58r"
val_seq = pad_sequences(val_input, maxlen = 100)

# %% id="ir_h8a5meqeJ"
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape = (100, 500)))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# %% id="C0M5IpW5fS1W"
train_oh = keras.utils.to_categorical(train_seq)

# %% colab={"base_uri": "https://localhost:8080/"} id="22cnkOfC4rTz" outputId="a25f895c-ccbc-461b-fc18-bcd5d38a2116"
print(train_oh.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="QYQO5-5Y4yGU" outputId="f8102cc4-93c7-4adf-af14-8bcdc631e294"
print(train_oh[0][0][:12])

# %% colab={"base_uri": "https://localhost:8080/"} id="wn5HRl4B406p" outputId="c9566b06-f231-4ae2-e920-75c86821974e"
print(np.sum(train_oh[0][0]))

# %% id="qiDZKD-jf0Ep"
val_oh = keras.utils.to_categorical(val_seq)

# %% colab={"base_uri": "https://localhost:8080/"} id="Eym3XJri5ABq" outputId="5e7be217-1824-49cc-aedf-dec3dfe27210"
model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="EGseWrJYf9XZ" outputId="beec3452-b634-42fa-c658-26dcf9b7039d"
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5')
earlystopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)
history = model.fit(train_oh, train_target, epochs = 100, batch_size = 64,
                    validation_data = (val_oh, val_target),
                    callbacks = [checkpoint_cb, earlystopping_cb])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="e5pfu0Pg5rEn" outputId="7f33649c-f93b-4c8c-a0d2-92f4a284791c"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="Re12qr1g5-9r" outputId="6070c8be-de2c-4059-8898-68e98fd1ff9e"
print(train_seq.nbytes, train_oh.nbytes)

# %% id="Z9_B_j1U6N6T"
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length = 100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation = 'sigmoid'))

# %% colab={"base_uri": "https://localhost:8080/"} id="w9RUARQ87WK4" outputId="d2e24ad9-6764-4ccd-f1fb-be23d2c7a66d"
model2.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="EeLTLfOJ8PB0" outputId="ae824d38-b302-4d6d-eb84-d2d77c114202"
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model2.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5')
earlystopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)
history = model2.fit(train_seq, train_target, epochs = 100, batch_size = 64,
                     validation_data = (val_seq, val_target),
                     callbacks = [checkpoint_cb, earlystopping_cb])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="Mwi2mTVD9SiT" outputId="374e1aad-72e7-4f9e-ca3f-ee029c01a349"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
