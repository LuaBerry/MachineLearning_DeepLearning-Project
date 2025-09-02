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

# %% id="MKsDdioEdJHn"
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# %% id="_aFOgWKEdK7w"
from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42)

# %% id="OOC3Xrycdta5"
dense1 = keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784,))
dense2 = keras.layers.Dense(10, activation='softmax')

# %% id="GmXhQjDSeaRw"
model = keras.Sequential([dense1, dense2])

# %% colab={"base_uri": "https://localhost:8080/"} id="XbjNyO9kepxY" outputId="925bc648-c188-40e5-d56a-2000b4ce7436"
model.summary()

# %% id="kKJxfeTceyY_"
model = keras.Sequential([
                          keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784,), name = 'hidden'),
                          keras.layers.Dense(10, activation = 'softmax', name = 'output')],
                          name = 'Fashion MNIST Model'
)

# %% colab={"base_uri": "https://localhost:8080/"} id="lGpBl-ahf6V1" outputId="3ffbabaa-3aab-44f5-8e62-4d1a7c280e73"
model.summary()

# %% id="fLYMJZC7f7u4"
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

# %% colab={"base_uri": "https://localhost:8080/"} id="QqIpeTDsgTdI" outputId="450d346b-8be3-4f8a-ece0-81a4ed66c5e3"
model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="fAuKFCYTgUpP" outputId="9469190b-6385-4b06-d1d4-ea293d06b70c"
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5)

# %% id="Dj8BOMe5gtjn"
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

# %% colab={"base_uri": "https://localhost:8080/"} id="Ry4y_9PihTh-" outputId="90a24311-1b1c-40f2-8933-7839f51f0178"
model.summary()

# %% id="IwO2JF6whWZQ"
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)

# %% colab={"base_uri": "https://localhost:8080/"} id="b4HKO1OWhzbi" outputId="ec08a11a-ac79-4321-f126-862f797be345"
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5)

# %% colab={"base_uri": "https://localhost:8080/"} id="qlzIps7uh-Jo" outputId="a500d8eb-e01e-4250-a57f-be072c50b25f"
model.evaluate(val_scaled, val_target)

# %% id="TLtjhVrziOYw"
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

# %% colab={"base_uri": "https://localhost:8080/"} id="nAPM8Lb8jN3w" outputId="6ec01afe-f9ee-4a6c-af49-4d9ab7f6196c"
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5)

# %% colab={"base_uri": "https://localhost:8080/"} id="fgza6AO3jZXS" outputId="e629dda5-d72d-4111-a6b1-53b9ffe71b39"
model.evaluate(val_scaled, val_target)
