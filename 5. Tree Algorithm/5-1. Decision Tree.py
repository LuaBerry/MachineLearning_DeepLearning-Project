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

# %% id="WvHUDso1BO3o"
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="fNL3cwigBXf8" outputId="2af74121-6475-4aaf-cef3-bc75b21f645c"
wine.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="LhMtBhgABY-L" outputId="dac49caa-80fb-4fd2-d272-b4e1d3a5521e"
wine.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="NBWNnlSHBmfT" outputId="59473389-59aa-4197-aba0-c1fba64b40f9"
wine.describe()

# %% id="mvTkqN-0BvS0"
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# %% id="iADofImvCE4U"
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_scaled = ss.fit_transform(train_input)
test_scaled = ss.transform(test_input)

# %% colab={"base_uri": "https://localhost:8080/"} id="9vYnw9MqCrEU" outputId="de4d9af3-83ee-4c30-f51d-46ead7f26b52"
print(train_scaled.shape, test_scaled.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="Oy56T_MyC-Dz" outputId="da285f16-ec72-4743-c448-2b8e141027ca"
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="rpFs61IeDRUN" outputId="7effe230-acdd-4ca8-841b-36d0d2c5630b"
print(lr.coef_, lr.intercept_)

# %% colab={"base_uri": "https://localhost:8080/"} id="D24wYa_qDlMT" outputId="d70abf9b-2932-4a36-e26c-73eb56719908"
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/", "height": 411} id="XDEMTfBTEWoz" outputId="456a129e-e2d5-483c-a236-5aa0d0cfdfaf"
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize = (10, 7))
plot_tree(dt)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 411} id="40YRhb5JEm_j" outputId="ad58cd52-f62e-4d82-bd39-bdc861a931d5"
plt.figure(figsize = (10, 7))
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="I8VaMuuNFiMe" outputId="96174808-9413-4e26-f462-69cd97f73a73"
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/", "height": 846} id="xbsOqSHtIM1t" outputId="bd1d23f4-2030-4640-efb7-42df79886614"
plt.figure(figsize = (20,15))
plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="g3qdWWxsIZJE" outputId="9a568c6b-5741-4305-c875-e6618daacce3"
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# %% colab={"base_uri": "https://localhost:8080/", "height": 846} id="pdhzlF4MI3pk" outputId="d1877ca7-59e6-4494-d42a-4e6512103bd3"
plt.figure(figsize = (20,15))
plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="fAoCvhL6JGUj" outputId="8d242127-886c-4e6c-adff-0f981997a0bf"
print(dt.feature_importances_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 883} id="WgHw2gFIMJYm" outputId="2ed1b971-7195-4aad-f580-a5b5941b2220"
dt = DecisionTreeClassifier(min_impurity_decrease = 0.0005, random_state = 42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
plt.figure(figsize = (20,15))
plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()
