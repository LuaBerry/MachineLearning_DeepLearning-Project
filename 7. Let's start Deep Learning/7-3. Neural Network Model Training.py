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

# %% colab={"base_uri": "https://localhost:8080/"} id="4yk1fbSrQGfC" outputId="a0d4f68b-2d28-4ef2-f257-42b0ebede587"
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target,
                                                                      test_size = 0.2, random_state = 42)


# %% id="8nmQiKVTQ612"
def model_fn(a_layer = None):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape = (28, 28)))
  model.add(keras.layers.Dense(100, activation = 'relu'))
  if a_layer:
    model.add(a_layer)
  model.add(keras.layers.Dense(10, activation = 'softmax'))
  return model


# %% colab={"base_uri": "https://localhost:8080/"} id="cHr-Gtm3RtE9" outputId="f73bd1c7-1a89-4f94-9388-32df3f0786f7"
model = model_fn()
model.summary()

# %% id="kDhHuQ46Ryp5"
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 5, verbose = 0)

# %% colab={"base_uri": "https://localhost:8080/"} id="p7borXOrSGMb" outputId="8f31e7fb-1a74-46ca-c692-16fc37a32ed4"
print(history.history.keys())

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="sZXNJmvESSLD" outputId="0ffb512a-da6b-4e5a-ef8f-043d6c090423"
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="pNE4sYY-SfHT" outputId="b8fb0f5c-bb73-47a7-9d6b-65efb6fdbe4e"
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="jYa3vs2DSpC6" outputId="7e80acb1-6181-43d9-b320-b5f34a72b366"
model = model_fn()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="QUexEHiFTLpo" outputId="68175195-3113-4eda-8094-ee10964b5949"
model = model_fn()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0,
                    validation_data = (val_scaled, val_target))
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="xUAhd0w4UK01" outputId="1e66a73b-3e01-4c8f-bd79-a2f10305b8ae"
print(history.history.keys())

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="3TfX-w3dUnzN" outputId="a6f70e91-81a4-46b1-a5c6-29bc8843a8bb"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="6GsDFIDrUtdP" outputId="629b7ea1-3838-4186-dfe2-0c13f073df85"
model = model_fn()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0,
                    validation_data = (val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="SYcuhLjcU8Lp" outputId="a9773dbf-8588-4070-9382-09d01ad6cced"
model = model_fn(keras.layers.Dropout(0.3))
model.summary()

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="Xr8eVibHVcjv" outputId="b3927617-7eed-432f-e40f-d0d6749f5811"
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0,
                    validation_data = (val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="mbN_XPilWV4p" outputId="f1561151-a90a-431b-f98e-00cf15d9b5d5"
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 10, validation_data = (val_scaled, val_target))

# %% id="BTfyOfBnW6y0"
model.save_weights('model_weights.h5')
model.save('model-whole.h5')

# %% colab={"base_uri": "https://localhost:8080/"} id="aYgelEwYXMfL" outputId="9963ab3c-ca16-44e2-ecad-2514e8982c0c"
# !ls -al *.h5

# %% id="n1wdSv7hXXw1"
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model_weights.h5')

# %% colab={"base_uri": "https://localhost:8080/"} id="V_8NelXuXorm" outputId="45e78328-c19e-41d5-c3bf-0773d08bcb41"
import numpy as np
val_labels = np.argmax(model.predict(val_scaled), axis = -1)
print(np.mean(val_labels == val_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="HEInAujnYCd8" outputId="4d2a9610-14eb-4dc5-e28c-9853de64d9e5"
model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="Fid7OHCiYdPX" outputId="8dd6e97c-8319-4f56-c2fd-62f8cd3438ff"
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_model.h5')
model.fit(train_scaled, train_target, epochs = 20,
          validation_data = (val_scaled, val_target),
          callbacks = [checkpoint_cb])

# %% colab={"base_uri": "https://localhost:8080/"} id="Se2vd616bPxQ" outputId="e4f35552-0011-49f0-d861-289d0dd80d6c"
model = keras.models.load_model('best_model.h5')
model.evaluate(val_scaled, val_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="hw0UwiUhfAFY" outputId="63ac6468-1e09-4a5a-c9be-5430eb9dd71a"
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)
history = model.fit(train_scaled, train_target, epochs = 20,
                    validation_data = (val_scaled, val_target),
                    callbacks = [checkpoint_cb, early_stopping_cb])

# %% colab={"base_uri": "https://localhost:8080/"} id="x8NLsyHpgO2I" outputId="bf435d27-7679-442c-bf72-79bddf7a3667"
print(early_stopping_cb.stopped_epoch)

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="Hy2RzzDuhViu" outputId="ec5fa86a-b318-42a7-d95f-9c2a03ef7382"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="U6PFCGJmhu7a" outputId="26ccd784-853f-4ef0-9601-4088341bc25a"
model.evaluate(val_scaled, val_target)

# %% id="NxegpsvIh3N6"
