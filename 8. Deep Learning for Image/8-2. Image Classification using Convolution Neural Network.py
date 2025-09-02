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

# %% colab={"base_uri": "https://localhost:8080/"} id="dNyMdIsYfS1Z" outputId="c1cc8652-b2ca-42bb-e63f-be6d652dee2c"
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target,
                                                                      test_size = 0.2, random_state = 42)

# %% id="gweCKtjxg1lU"
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu',
                             padding = 'same', input_shape = (28, 28, 1)))

# %% id="LLOUi2T9hti8"
model.add(keras.layers.MaxPooling2D(2))

# %% id="wpHfOf-sh9K4"
model.add(keras.layers.Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(keras.layers.MaxPooling2D(2))

# %% id="3rOxTZbJiXVd"
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation = 'softmax'))

# %% colab={"base_uri": "https://localhost:8080/"} id="3oOvEDgzigX3" outputId="716f4ccb-272b-4b97-96e0-fb4399797810"
model.summary()

# %% colab={"base_uri": "https://localhost:8080/", "height": 856} id="zXgrhQ1Oi4r_" outputId="d7b0cf3a-2c90-49f9-fca7-fe19b24bd5ee"
keras.utils.plot_model(model)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1vSnoCPHjMbN" outputId="155c69b9-dda7-407b-c846-81775613ac45"
keras.utils.plot_model(model, show_shapes = True,
                       to_file = 'cnn-architecture.png', dpi = 300)

# %% colab={"base_uri": "https://localhost:8080/"} id="8q5Au94GjkJN" outputId="68872ec0-aa6b-4f83-8e9e-f10de60ad927"
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
earlystopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)

history = model.fit(train_scaled, train_target, epochs = 20,
                    validation_data = (val_scaled, val_target),
                    callbacks = [checkpoint_cb, earlystopping_cb])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="78QlejGck4Nb" outputId="e56c7d91-286c-48ee-e064-ecdb834a29e5"
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="P2rZiE8gl6Di" outputId="dcc51e9e-2d06-47e9-811a-e03f83fd95ac"
model.evaluate(val_scaled, val_target)

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="Xp39T7M0mAKf" outputId="08acc4b8-55c5-4ec8-b75f-655de602460e"
plt.imshow(val_scaled[0].reshape(28, 28), cmap = 'gray_r')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="dO629NxOmMKP" outputId="3747a236-e7e1-4a34-e2c8-fc8f09ab92e1"
preds = model.predict(val_scaled[0:1])
print(preds)

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="wAg8JgRPmSRf" outputId="0d0be063-0bf8-4b9a-dd3e-796c013ab8e3"
plt.bar(range(1,11), preds[0])
plt.xlabel('class')
plt.ylabel('probability')
plt.show()

# %% id="7k86KR2SmqUX"
classes = ['T-Shirt', 'Pants', 'Sweater', 'Dress', 'Coat', 'Sandles', 'Shirt', 'Sneakers', 'Bag', 'Ankle Boots']

# %% colab={"base_uri": "https://localhost:8080/"} id="x5nvtM_Emi6O" outputId="03186af2-84fe-4b8a-cdd8-723d2f996035"
import numpy as np
print(classes[np.argmax(preds)])

# %% id="jjARjONfnSAz"
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

# %% colab={"base_uri": "https://localhost:8080/"} id="XW2jkvcnngtd" outputId="80d912bd-d156-41f1-b746-afe6f8fc96ac"
model.evaluate(test_scaled, test_target)

# %% id="wL4OTjpHnkRL"
