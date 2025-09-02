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

# %% id="J9N8mEyv0izw"
import numpy as np

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="pUb0sjYI0jtl" outputId="7e12ccb9-cdd8-49f5-85b7-d890f82dd2e1"
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()

# %% id="SvVE0XJq1FX8"
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)

# %% colab={"base_uri": "https://localhost:8080/"} id="zSdZ_Woe1n8M" outputId="5dbe10b1-f1f9-4b7f-9f99-c97f77684db2"
test_array = np.array([1,2,3,4])
print(test_array.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="sJ1HIGmU17T1" outputId="5a9d07e3-2b66-4044-9764-e5881067875e"
test_array = test_array.reshape(2, 2)
print(test_array.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="DFwk612F2CS0" outputId="a72cc862-4c6c-4847-af22-cdd461dcca89"
print(test_array)

# %% id="_X5yNRfo2EU8"
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="SlDf3Rgp2k7E" outputId="968a85e9-3e1d-4a2d-dba7-a758e5e7f64a"
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="YtkY_Yxw3I5N" outputId="40c08887-7a92-4086-f195-27710f6d61f7"
print(knr.score(test_input, test_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="xoG0wZL83NBc" outputId="ea03ca12-7ff3-4926-d1ad-864da6ba012c"
from sklearn.metrics import mean_absolute_error
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

# %% colab={"base_uri": "https://localhost:8080/"} id="XkhAtzxr39n9" outputId="925eb317-789d-4b89-c41b-5c772ca0e36a"
print(knr.score(train_input, train_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="x9rBA8Z64jnM" outputId="e158fb5c-64b3-41a0-cb75-f1536d6869c8"
knr.n_neighbors = 3

knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="eFV4mL8w6EzN" outputId="6a7d10cb-5c49-4805-b4b8-943e7609a196"
print(knr.score(test_input, test_target))

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Jzh9ITYh6H79" outputId="bfd06bcf-39d4-4d0e-8094-334bdfaa245f"
x = np.arange(5, 45).reshape(-1, 1)

for n in [1, 3, 5, 10, 25]:
  knr.n_neighbors = n
  knr.fit(train_input, train_target)
  prediction = knr.predict(x)

  plt.scatter(train_input, train_target)
  plt.plot(x, prediction)
  plt.title('n_neighbors = {}'.format(n))
  plt.xlabel('length')
  plt.ylabel('weight')
  plt.show()

# %% id="pl56Lhmx8knW"
