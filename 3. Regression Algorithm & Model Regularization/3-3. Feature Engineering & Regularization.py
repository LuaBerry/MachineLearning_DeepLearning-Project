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

# %% id="c60LBtyVUUs0"
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

# %% id="VricINCdUxyp"
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# %% id="6QwhYTlqVP3M"
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42)

# %% id="61ToXUtDV7j-"
from sklearn.preprocessing import PolynomialFeatures

# %% colab={"base_uri": "https://localhost:8080/"} id="jUA5QTPhWOdb" outputId="d23ba277-f757-4588-bf1d-1d1a2052a39c"
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))

# %% colab={"base_uri": "https://localhost:8080/"} id="U3BCNQb9Wclu" outputId="c48e45a0-8ecb-49d3-e60d-09d8733da92a"
poly = PolynomialFeatures(include_bias = False)
print(poly.fit_transform([[2, 3]]))

# %% id="hlZLfv_xW6P9"
poly = PolynomialFeatures(include_bias = False)
train_poly = poly.fit_transform(train_input)
print(train_poly.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="5rO4gefCXLNW" outputId="02df045c-2b25-43a2-818a-ebe8047935d4"
poly.get_feature_names_out()

# %% id="rK6FMgrhXTPs"
test_poly = poly.transform(test_input)

# %% colab={"base_uri": "https://localhost:8080/"} id="K91Z7fIMXnFl" outputId="4406a583-f02e-4dfb-900a-ba651b272820"
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# %% id="RxvnxilfX3QN"
poly = PolynomialFeatures(include_bias = False, degree = 5)
train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)

# %% id="1ICkJpq2Ye4N"
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))

# %% id="FOkrV6n5YpeN"
print(lr.score(test_poly, test_target))

# %% id="dMvfrkciYxuV"
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_scaled = ss.fit_transform(train_poly)
test_scaled = ss.transform(test_poly)

# %% id="bErG2LjHZ2xP"
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# %% id="57cPuGnMaOX0"
import matplotlib.pyplot as plt
train_score = []
test_score = []

# %% id="qYfcFaoPadYU"
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for a in alpha_list:
  ridge = Ridge(alpha = a)
  ridge.fit(train_scaled, train_target)
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="AxoH6QgWa2w2" outputId="2eb0ee2e-5b3f-4426-b48b-e3ab1707f6dc"
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('log10 alpha')
plt.ylabel('R^2')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="7CPR3dxlbL1d" outputId="9243645e-93ed-4945-abd2-89295c3ad4e4"
ridge = Ridge(alpha = 0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="-iUmQOv8bUp9" outputId="7e28bfd0-75a4-4c37-f77c-c92081eb1f18"
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# %% id="XFE2BSgNbcxd"
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
train_score = []
test_score = []
for a in alpha_list:
  lasso = Lasso(alpha = a, max_iter=10000)
  lasso.fit(train_scaled, train_target)
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))

# %% id="nkwsgNVAbwL5"
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('log10 alpha')
plt.ylabel('R^2')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="d4odyvKWcBvJ" outputId="0ecc7331-3ad8-43f3-8a6d-0e485138d57b"
lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="D8cciCF2cKmF" outputId="72f65427-01d3-420f-ab5c-fe63438c399e"
print(np.sum(lasso.coef_ == 0))

# %% id="8t81E84scRZd"
