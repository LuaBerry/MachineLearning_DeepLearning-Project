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

# %% id="ekkSp_0YO5tG"
# !wget https://bit.ly/fruits_300_data -O fruits_300.npy
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)

# %% colab={"base_uri": "https://localhost:8080/"} id="1gKiEDL7SxAU" outputId="39484057-c6da-4828-b583-00fce7e62205"
from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
pca.fit(fruits_2d)

# %% colab={"base_uri": "https://localhost:8080/"} id="gq9Rby_bTKfX" outputId="d2de7b7f-f0d4-41fa-e676-0506a82d6cda"
print(pca.components_.shape)

# %% id="TZ1rlLhPTh1s"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 303} id="y3AAlJo8TOdV" outputId="190c54e0-fdf7-4e37-d74d-3a5480e915d0"
draw_fruits(pca.components_.reshape(-1, 100, 100))

# %% colab={"base_uri": "https://localhost:8080/"} id="iyns8XttToMv" outputId="d34e0673-6b7b-4c6c-f964-53fe74298cd2"
print(fruits_2d.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="WscS56psT3tn" outputId="791a9047-8b23-437f-b776-b872b1aba495"
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="8Wiwr97tT9Hn" outputId="750ff96c-6f92-48b0-8a79-a5d0688f0b5a"
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="RXesV5GPULGf" outputId="4bf0ae3d-d3ff-440b-b61f-788f9e09a85c"
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
  draw_fruits(fruits_reconstruct[start:start+100])
  print("\n")

# %% colab={"base_uri": "https://localhost:8080/"} id="3_04-z--Ua2w" outputId="8ff36fb5-6b8a-4041-abe5-34f0a85fb52a"
print(np.sum(pca.explained_variance_ratio_))

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="FtyOfO94U1__" outputId="67cfd010-aa90-4071-d9c6-916680a6c29a"
plt.plot(pca.explained_variance_ratio_)
plt.show()

# %% id="-D8owgTpU9AX"
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
target = np.array([0]* 100 + [1] * 100 + [2] * 100)

# %% colab={"base_uri": "https://localhost:8080/"} id="C1iL6VunVSW3" outputId="6bf6155e-27b0-4b59-b5ec-2571f2715f43"
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# %% colab={"base_uri": "https://localhost:8080/"} id="RB8M6JBiVgt3" outputId="1d6de573-cb11-467e-a53f-2fbfdedcd383"
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# %% colab={"base_uri": "https://localhost:8080/"} id="n6H4KFzBV3FK" outputId="62006de1-e6db-40f1-fe89-28d128ccefe9"
pca = PCA(n_components = 0.5)
pca.fit(fruits_2d)

# %% colab={"base_uri": "https://localhost:8080/"} id="AqKbA8TcWFkz" outputId="a9c5a887-590d-41a9-d416-a6f690a90587"
print(pca.n_components_)

# %% colab={"base_uri": "https://localhost:8080/"} id="7RMeK_jDWJm-" outputId="c8d3daab-2e5d-492b-e2a2-565f947eddab"
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="0BFcWF-OWP_u" outputId="3bfd828d-1320-4983-8591-37d8cd6c412e"
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# %% colab={"base_uri": "https://localhost:8080/"} id="K3SUoFNtWSqW" outputId="0a4d8caf-3d79-43cd-ff38-1604f325f57a"
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts = True))

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="N2BpVFLjWm5A" outputId="a151f151-771a-472a-968b-68fa974ff932"
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
print('\n')

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="5Vhy8Td9XC6I" outputId="6b1a7270-299b-4d11-9176-fb472452f35d"
for label in range(0, 3):
  data = fruits_pca[km.labels_ == label]
  plt.scatter(data[:, 0], data[:, 1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
