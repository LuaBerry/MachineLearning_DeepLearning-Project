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

# %% id="cCs38FA9D_P9"
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

# %% id="XsPaX2KYEG1z"
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)

# %% id="5pppN-q_EZG9"
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

# %% colab={"base_uri": "https://localhost:8080/"} id="ZGRWWM9CEgCS" outputId="ec33e8fd-f445-4b16-d114-52184f328be7"
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.n_neighbors = 3
knr.fit(train_input, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="aZuYiQjbExrN" outputId="8f033187-99f2-4160-a6e3-16654774699a"
print(knr.predict([[50] ] ))

# %% colab={"base_uri": "https://localhost:8080/", "height": 297} id="DM22FkOxE3_b" outputId="03ff8a8f-63dd-4fe1-ea92-5697444dae71"
import matplotlib.pyplot as plt

distances, indexes = knr.kneighbors([[50]])

plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.scatter(50, 1033, marker = '^')
plt.xlabel("length")
plt.ylabel("weight")

# %% colab={"base_uri": "https://localhost:8080/"} id="viR_Nwk4F6zK" outputId="366543d4-e870-45de-fb9b-5b2c8cb1882d"
print(np.mean(train_target[indexes]))

# %% colab={"base_uri": "https://localhost:8080/"} id="vB7Ebha6Grrq" outputId="f80a3569-11e7-459b-941e-d7e96149d6d8"
print(knr.predict([[100]]))

# %% colab={"base_uri": "https://localhost:8080/", "height": 297} id="ty8QaofPGwCy" outputId="d6e2bad8-2db4-42fa-8233-8656efc13e1a"
import matplotlib.pyplot as plt

distances, indexes = knr.kneighbors([[50]])

plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.scatter(100, 1033, marker = '^')
plt.xlabel("length")
plt.ylabel("weight")

# %% colab={"base_uri": "https://localhost:8080/"} id="EruZ-Qc2G0uK" outputId="bb0cdba2-91b0-4c27-afcb-48783f7f5816"
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(train_input, train_target)

print(lr.predict([[50]]))

# %% colab={"base_uri": "https://localhost:8080/"} id="nmT9bmzKHiKC" outputId="2d622f76-2372-440e-9df8-33d02effa858"
print(lr.coef_, lr.intercept_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="fgtfsOTEHrqS" outputId="6d8b0f84-10cf-4bc9-b76b-0890c2cc25ff"
plt.scatter(train_input, train_target)

plt.plot([15,50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])

plt.scatter(50, 1241.8, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="r0wfpGmiJKla" outputId="8b1adb81-0df9-4d29-e243-9208571d0f91"
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

# %% id="SI-cnQhnJdCi"
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# %% colab={"base_uri": "https://localhost:8080/"} id="bvHuBtouKcbD" outputId="89c80a8a-0cf2-47de-9190-0cb132b378ce"
print(train_poly.shape, test_poly.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="nWSmrKCCKgKb" outputId="60394383-56c7-4477-fb2d-730cf405bf4b"
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))

# %% colab={"base_uri": "https://localhost:8080/"} id="Mz977vneKz97" outputId="a647caaf-d29a-42b5-a89a-9fa643a76acc"
print(lr.coef_, lr.intercept_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="ibgVAefqLsCD" outputId="a8486e31-b285-45ec-cd5e-4f07441e3284"
point = np.arange(15, 50)
plt.scatter(train_input, train_target)

plt.plot(point, 1.01*(point**2) - 21.6 * point + 116.05)

plt.scatter(50, 1574, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="jSkAA_wQM42b" outputId="806a0c9c-9f7c-4614-c72e-934d6799bde0"
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# %% id="ZyW7c8awNuYr"
