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

# %% id="zPUiKj96E-9g"
# !wget https://bit.ly/fruits_300_data -O fruits_300.npy

# %% id="gPW59yqhFIN1"
import numpy as np
import matplotlib.pyplot as plt

# %% id="qqRMGn7eFYD9"
fruits = np.load('fruits_300.npy')

# %% colab={"base_uri": "https://localhost:8080/"} id="5B1YzrRuFfAr" outputId="491c11e5-252f-42a7-d1e2-cb7188ed91d8"
print(fruits.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="YXoynYBkFiL7" outputId="3c480967-a1f9-4b08-b116-511d60787563"
print(fruits[0, 0, :])

# %% colab={"base_uri": "https://localhost:8080/", "height": 268} id="p2k58pR_Fpls" outputId="0a23f0a2-8328-43a9-b901-c940dc1ae95d"
plt.imshow(fruits[0], cmap = 'gray')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 268} id="dGjbhNKiF3DU" outputId="6d24b295-b40b-4f73-a5cb-f284a6b8b24a"
plt.imshow(fruits[0], cmap = 'gray_r')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 203} id="G-7FVfY3GLA7" outputId="a2ffd2d1-5699-4b2c-cf82-314177bb2450"
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap = 'gray_r')
axs[1].imshow(fruits[200], cmap = 'gray_r')
plt.show()

# %% id="Mw91FfuXGe-D"
apple = fruits[0:100].reshape(-1, 100 * 100)
pineapple = fruits[100:200].reshape(-1, 100 * 100)
banana = fruits[200:300].reshape(-1, 100 * 100)

# %% colab={"base_uri": "https://localhost:8080/"} id="3pJGfDH-HVa8" outputId="72fe968e-ac01-4d63-a91a-c9bc12e712af"
print(apple.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="Lv3CkoonHYGL" outputId="ccfebb50-ab66-4d8b-efb4-3eeaa32044e5"
print(apple.mean(axis = 1))

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="6odeqNTBHw8i" outputId="0abea74c-6383-41d3-f3b8-94408f500ad4"
plt.hist(np.mean(apple, axis = 1), alpha = 0.8)
plt.hist(pineapple.mean(axis = 1), alpha = 0.8)
plt.hist(banana.mean(axis = 1), alpha = 0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="VhnLD5OHIc-c" outputId="f95a3c76-80c6-421a-f566-0f4aba152e01"
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
axs[0].bar(range(10000), apple.mean(axis = 0))
axs[1].bar(range(10000), pineapple.mean(axis = 0))
axs[2].bar(range(10000), banana.mean(axis = 0))
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 322} id="kQfMM4ThI6tt" outputId="b4ce1e05-68af-49f5-fbca-59e26795ca77"
apple_mean = apple.mean(axis = 0).reshape(100, 100)
pineapple_mean = pineapple.mean(axis = 0).reshape(100, 100)
banana_mean = banana.mean(axis = 0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
axs[0].imshow(apple_mean, cmap = 'gray_r')
axs[1].imshow(pineapple_mean, cmap = 'gray_r')
axs[2].imshow(banana_mean, cmap = 'gray_r')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="QXPezHo_Kh-1" outputId="5b3643bd-601e-426e-b9cc-57d546742f85"
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1,2))
print(abs_mean.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 574} id="BZqJYs0iLTEt" outputId="1b93bccc-73e6-4c8f-9c20-b2e98b366f03"
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap = 'gray_r')
    axs[i, j].axis('off')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="q1wpyiAnL20d" outputId="3ff0c819-7936-49e5-955b-1d69c691f8f5"
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis = (1,2))
print(abs_mean.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 574} id="LkbYURtQNDfM" outputId="403777a3-c22a-4832-e91d-3730c9963624"
banana_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[banana_index[i*10 + j]], cmap = 'gray_r')
    axs[i, j].axis('off')
plt.show()
