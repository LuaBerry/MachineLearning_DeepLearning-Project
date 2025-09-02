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

# %% colab={"base_uri": "https://localhost:8080/"} id="IAMRSYH_WHkf" outputId="c759db36-6837-499b-b4d5-28b51842f5f8"
# !wget https://bit.ly/fruits_300_data -O fruits_300.npy

# %% id="lwuYIt5nWOk3"
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# %% colab={"base_uri": "https://localhost:8080/"} id="ifRbFNpUWXjV" outputId="3164d996-dc17-49c5-d51f-5a660fd02c67"
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_2d)

# %% colab={"base_uri": "https://localhost:8080/"} id="pksekg-RWfad" outputId="10fc5285-4e56-4dca-a81e-fda3f805fd82"
print(km.labels_)

# %% colab={"base_uri": "https://localhost:8080/"} id="aAtzETMHW2bm" outputId="7fa7a6e1-f433-4afa-d20f-f7bfb00a5958"
print(np.unique(km.labels_, return_counts = True))

# %% id="cu-5Pp5bW-zO"
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio = 1):
  n = len(arr)
  rows = int(np.ceil(n / 10))
  cols = n if rows < 2 else 10
  fig, axs = plt.subplots(rows, cols,
                          figsize = (cols * ratio, rows * ratio), squeeze = False)
  for i in range(rows):
    for j in range(cols):
        if i*10 + j < n:
          axs[i, j].imshow(arr[i * 10 + j], cmap = 'gray_r')
        axs[i, j].axis('off')
  plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 683} id="G537imCtYfk-" outputId="dd540d94-133f-427b-8797-ce4b377ba486"
draw_fruits(fruits[km.labels_ == 0])

# %% colab={"base_uri": "https://localhost:8080/", "height": 574} id="pUBAlV04X2C3" outputId="7ad43567-7b6e-4f01-8668-ba096d9f21bb"
draw_fruits(fruits[km.labels_ == 1])

# %% colab={"base_uri": "https://localhost:8080/", "height": 574} id="xrUrIV_QZMHf" outputId="faedcf4b-98b2-410e-ee04-93b8c9499777"
draw_fruits(fruits[km.labels_ == 2])

# %% colab={"base_uri": "https://localhost:8080/", "height": 179} id="plQdZuLlZOT-" outputId="7d2e19e6-9410-4077-b88d-6fb7cee7edf2"
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio = 3)

# %% colab={"base_uri": "https://localhost:8080/"} id="pNQ0loahZrvW" outputId="75322ab4-f5be-47d1-ba41-1e7555d31223"
print(km.transform(fruits_2d[100:101]))

# %% colab={"base_uri": "https://localhost:8080/"} id="Lvq07tP2aA-v" outputId="fa7e3cf3-13d2-456f-d86a-5f3c3fa305e4"
print(km.predict(fruits_2d[100:101]))

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} id="i0-lYtqyaOf2" outputId="cfc64ca1-6399-4299-8e10-2975fcd5d26e"
draw_fruits(fruits[100:101])

# %% colab={"base_uri": "https://localhost:8080/"} id="1JuYTi-6aUH-" outputId="be46781c-e47e-4359-ccf7-2de628c57f3e"
print(km.n_iter_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 290} id="MHdmKSV8aYuO" outputId="ee34d951-8b8c-4dec-9c82-5b96b629b386"
inertia = []
for k in range(2, 7):
  km = KMeans(n_clusters = k, random_state = 42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
