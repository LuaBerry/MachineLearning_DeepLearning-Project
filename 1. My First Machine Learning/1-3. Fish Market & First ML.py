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

# %% executionInfo={"elapsed": 296, "status": "ok", "timestamp": 1650161648891, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="Zn4jsNQpPnqr"
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# %% id="SKt87rrITGVa"
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% executionInfo={"elapsed": 296, "status": "ok", "timestamp": 1650161814910, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="d7_Nh3x7TK_2"
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# %% id="KY1cdklETSgh"
plt.scatter(bream_length, bream_weight,c = 'r')
plt.scatter(smelt_length, smelt_weight, c = 'b')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% executionInfo={"elapsed": 309, "status": "ok", "timestamp": 1650161968784, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="UqDXzsVzT_2d"
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# %% executionInfo={"elapsed": 323, "status": "ok", "timestamp": 1650162003968, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="JcrjvZ1yUShE"
fish_data = [[l, w] for l, w in zip(length, weight)]

# %% id="zMtgBek-Uq5A"
print(fish_data)

# %% id="zbUOy_3xU-w1"
fish_target = [1] * 35 + [0] * 14
print(fish_target)

# %% executionInfo={"elapsed": 302, "status": "ok", "timestamp": 1650162230347, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="WdICXoRwVL4n"
from sklearn.neighbors import KNeighborsClassifier

# %% executionInfo={"elapsed": 300, "status": "ok", "timestamp": 1650162250480, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="w_sqOzDwVW9j"
kn = KNeighborsClassifier()

# %% id="oQHK00X8Vj9B"
kn.fit(fish_data, fish_target)

# %% id="bSe_PHdCVomj"
kn.score(fish_data, fish_target)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 362, "status": "ok", "timestamp": 1650162589108, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="rPEgU1aoWXjg" outputId="fa2eb033-d0be-4d87-a233-35094fca5564"
kn.predict([[12, 5], [12, 100], [30, 600], [25, 100], [12, 800], [40, 5]])

# %% id="RAbQUM_yXJNy"
print(kn._fit_X)

# %% id="Hq2dkhdEXNze"
print(kn._y)

# %% executionInfo={"elapsed": 333, "status": "ok", "timestamp": 1650162870757, "user": {"displayName": "\ub77c\uc988\ubca0\ub9ac\ubc88\uc5ed\uac00", "userId": "15925344931567563578"}, "user_tz": -540} id="w1pEZbsgXgWH"
kn49 = KNeighborsClassifier(n_neighbors=49)

# %% id="QG6oyveDXmtl"
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

# %% id="zYwXQurnX22R"
print(35/49)

# %% id="d19Lm2X1ZvO7"
for i in range (5, 50):
  kn.n_neighbors = i
  score = kn.score(fish_data, fish_target)
  print(i, score)

