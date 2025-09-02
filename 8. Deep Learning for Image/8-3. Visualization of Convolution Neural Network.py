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

# %% id="8jrw6bUWvyNy"
from tensorflow import keras
model = keras.models.load_model('best-cnn-model.h5')

# %% colab={"base_uri": "https://localhost:8080/"} id="y1WsOucBwN2I" outputId="342d68e4-e79a-466d-94f5-278946518959"
model.layers

# %% colab={"base_uri": "https://localhost:8080/"} id="Sue7XoRfwZDz" outputId="c3e13825-8619-4072-d056-8a55ee7766d1"
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="8KNGrFQDyYQK" outputId="91c0b300-1112-4a01-a901-f028829920d4"
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())

# %% colab={"base_uri": "https://localhost:8080/", "height": 357} id="cO9CO32O-5AP" outputId="f1eca634-d50f-457e-b4b5-08041a3e48c4"
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 134} id="z82-skUz_NyI" outputId="bd5916db-ffbd-4617-a9f0-b86a66a5edf4"
fig, axs = plt.subplots(2, 16, figsize = (15, 2))
for i in range(2):
  for j in range(16):
    axs[i, j].imshow(conv_weights[:, :, 0, i * 16 + j], vmin = -0.5, vmax = 0.5)
    axs[i, j].axis('off')
plt.show()

# %% id="u_8PZcJjAPiM"
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))

# %% colab={"base_uri": "https://localhost:8080/"} id="3uwCChdZArQT" outputId="cfbde134-de96-4811-e766-f7bc55bb58f4"
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="boHSZ-UHBi4j" outputId="e4f7420e-fae5-4f46-c52f-f04017b8a7f4"
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="65WFxT0GCVfR" outputId="cc3099c9-fd5c-4c2a-c089-f6ac7911b0c2"
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weights')
plt.ylabel('count')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 134} id="u8ILKYQiChKQ" outputId="ad207682-9ee1-4db1-db44-24fa2195ffdc"
fig, axs = plt.subplots(2, 16, figsize = (15, 2))
for i in range(2):
  for j in range(16):
    axs[i, j].imshow(no_training_weights[:, :, 0, i * 16 + j], vmin = -0.5, vmax = 0.5)
    axs[i, j].axis('off')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="8aVSFLx9CpRf" outputId="bdd586cb-f415-4a52-8164-ef16838ff9b9"
print(model.input)

# %% id="UwwnBVj5DaJ_"
conv_acti = keras.Model(model.input, model.layers[0].output)

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} id="wdw4Dq4hEBOK" outputId="1bce950b-6ffa-4500-ac71-b73c14d1828b"
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap = 'gray_r')
plt.show()

# %% id="r5aqbtBIEOxx"
inputs = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_acti.predict(inputs)

# %% colab={"base_uri": "https://localhost:8080/"} id="LWJTE3nVEh2c" outputId="6efc088c-34b3-4cb5-852a-cb356eff60b3"
print(feature_maps.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 460} id="ps60VlVXEj9Y" outputId="40a07638-ee8d-4aaf-9559-63da897ee3e8"
fig, axs = plt.subplots(4, 8, figsize = (15, 8))
for i in range(4):
  for j in range(8):
    axs[i, j].imshow(feature_maps[0, :, :, i * 8 + j])
    axs[i, j].axis('off')
plt.show()

# %% id="LJo6hxXkFM2f"
conv2_acti = keras.Model(model.input, model.layers[2].output)

# %% id="B4TLTLjpFcpW"
feature2_maps = conv2_acti.predict(inputs)

# %% colab={"base_uri": "https://localhost:8080/"} id="cIK43P5kFjzc" outputId="32022ff9-c870-489d-dfe4-c0fbdee1c105"
print(feature2_maps.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 683} id="5Fz-1-KkFnCd" outputId="b0136567-b59a-4b52-b3e2-36af0e961135"
fig, axs = plt.subplots(8, 8, figsize = (12, 12))
for i in range(8):
  for j in range(8):
    axs[i, j].imshow(feature2_maps[0, :, :, i * 8 + j])
    axs[i, j].axis('off')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="C9o2JnGSF9kx" outputId="5058536f-b5ab-43b4-acaa-765371de7af7"
model.layers[2]
