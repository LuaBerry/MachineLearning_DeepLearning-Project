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

# %% colab={"base_uri": "https://localhost:8080/"} id="MNi6CpWR5pYl" outputId="56bd89c0-b0f5-4fb1-a870-615afdc5ae1a"
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# %% colab={"base_uri": "https://localhost:8080/"} id="PoNL9eEqS-8j" outputId="a89135a1-7409-469d-93aa-b48a34d3f09f"
print(train_input.shape, train_target.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="oOzLkdKYTVS-" outputId="1995839e-587e-475a-e941-ee672560e1bd"
print(test_input.shape, test_target.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 78} id="X7DyP-0aTak-" outputId="99770d18-ff05-4aef-be73-47d8ae4c790c"
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize = (10, 10))
for i in range (10):
  axs[i].imshow(train_input[i], cmap = 'gray_r')
  axs[i].axis('off')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="2blyZ0ZYTvqX" outputId="2787611a-8059-4eaa-f7e8-3351c89a4560"
print([train_target[i] for i in range(10)])

# %% colab={"base_uri": "https://localhost:8080/"} id="lYK979JKT05u" outputId="d8378e4e-de6a-428c-f27a-d9ddc6d01fb5"
import numpy as np
print(np.unique(train_target, return_counts = True))

# %% id="4GngQ96JUDhG"
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

# %% colab={"base_uri": "https://localhost:8080/"} id="QVzaYKG6UMf2" outputId="d899605c-2997-4481-a32b-258b119434dc"
print(train_scaled.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="fY4QogNZUO4-" outputId="ffcf76e4-d6ea-4f54-e75f-1e8f85ae3125"
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
sc = SGDClassifier(loss = 'log', random_state = 42, max_iter = 5)
scores = cross_validate(sc, train_scaled, train_target, n_jobs = -1)
print(np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="AYzZOvqnU1if" outputId="57770e0b-9285-4a81-dd86-d60d2a251908"
sc = SGDClassifier(loss = 'log', random_state = 42, max_iter = 10)
scores = cross_validate(sc, train_scaled, train_target, n_jobs = -1, return_train_score = True)
print(np.mean(scores['train_score']))
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# %% id="jtrmzTplVP2Q"
import tensorflow as tf

# %% id="AniGqrkIVmvG"
from tensorflow import keras

# %% id="ZHemtsWuVpQ-"
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42)

# %% colab={"base_uri": "https://localhost:8080/"} id="eJdl6iQQWDaY" outputId="b1517809-b92c-418a-a7cf-b3669d397593"
print(train_scaled.shape, train_target.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="U3BTlMrDWeXY" outputId="3df75a82-53f0-4e78-a238-85feca0dfcb9"
print(val_scaled.shape, val_target.shape)

# %% id="ekNXcVIcWzwz"
dense = keras.layers.Dense(10, activation = 'softmax', input_shape = (784,))

# %% id="M0ZnSTcgXCTj"
model = keras.Sequential(dense)

# %% id="2asSgP1nXE0G"
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')

# %% colab={"base_uri": "https://localhost:8080/"} id="86Ekq94LYG5u" outputId="fe29eacf-8c8e-4540-e562-c0f6f3395f3a"
print(train_target[:10])

# %% colab={"base_uri": "https://localhost:8080/"} id="5jae7clhYwt3" outputId="8ac95f61-a602-4125-c2c7-c75b5e182b3a"
model.fit(train_scaled, train_target, epochs = 5)

# %% colab={"base_uri": "https://localhost:8080/"} id="6TnKoKANY--v" outputId="c6d261b1-fbca-4118-8be7-02000a115032"
model.evaluate(val_scaled, val_target)
