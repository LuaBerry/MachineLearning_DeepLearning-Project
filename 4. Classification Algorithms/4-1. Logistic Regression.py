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

# %% id="rq9WJ7ZZ6jts"
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()

# %% id="Ya2UQqVr6r47"
print(pd.unique(fish['Species']))

# %% id="HtScW19X69fJ"
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

# %% id="GlMjQBxN7YLa"
print(fish_input[:5])

# %% id="mJvcMkOr7apB"
fish_target = fish[['Species']].to_numpy()

# %% id="aTkKnpse7gOZ"
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# %% id="LkkKTg_n8RTY"
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# %% id="QRp12hLc8khN"
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# %% id="gA3mgm4n853b"
print(kn.classes_)

# %% id="2BZpfnLb9RZZ"
print(kn.predict(test_scaled[:5]))
print(test_target[:5])

# %% id="hOhxv6lU9cyZ"
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 4))

# %% id="5dnANG6198L6"
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="y-8n-qOp_j7r" outputId="719dd859-4231-432f-eb6c-ecfd9cbb8939"
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# %% id="A6WVyNFfAPb7"
char_arr = np.array(['a', 'b', 'c', 'd', 'e'])
print(char_arr[[True, False, True, False,True]])

# %% id="AH2uwKMAC6iT"
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes[:,0]]
target_bream_smelt = train_target[bream_smelt_indexes]

# %% colab={"base_uri": "https://localhost:8080/"} id="XZ7WCP8FGhDd" outputId="6ff40e70-7adf-4f6d-bc9e-15c4765a0c92"
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# %% colab={"base_uri": "https://localhost:8080/"} id="YxLE4XRRGqhE" outputId="2f749e7d-9df6-4447-fdb5-71afbafb7758"
print(lr.predict(train_bream_smelt[:5]))

# %% colab={"base_uri": "https://localhost:8080/"} id="NBzdjMBGGuxT" outputId="0f19e439-56b4-48ab-cade-ea846781971e"
print(lr.predict_proba(train_bream_smelt[:5]))

# %% colab={"base_uri": "https://localhost:8080/"} id="ObXSR1ywG3Jj" outputId="e9388733-cee7-4982-fff1-3b1b54a06a81"
print(lr.classes_)

# %% colab={"base_uri": "https://localhost:8080/"} id="wIBrleANHAmj" outputId="7861af8c-227a-4cec-84cd-8dff45fab1f0"
print(lr.coef_, lr.intercept_)

# %% colab={"base_uri": "https://localhost:8080/"} id="tU7O6c1nHOpj" outputId="5815b804-f29f-4479-8ec7-8a5d43be0edb"
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# %% colab={"base_uri": "https://localhost:8080/"} id="5LFsbW8GHXij" outputId="0a2c63e5-014e-4362-e9af-f82861abf3cb"
from scipy.special import expit
print(expit(decisions))

# %% colab={"base_uri": "https://localhost:8080/"} id="uIk5-1f0Hhsj" outputId="4df5726e-34b2-46c6-f0d0-b1272bf0e4ab"
lr = LogisticRegression(C = 20, max_iter = 1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="-m4tIWlmKCNV" outputId="e5623d76-4140-4131-8fb0-ea38be37704a"
print(lr.predict(test_scaled[:5]))
print(test_target[:5])

# %% colab={"base_uri": "https://localhost:8080/"} id="-7ZIztksKKV7" outputId="a23c496b-de3d-4eb9-b5f9-38e34b4ba420"
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 3))

# %% colab={"base_uri": "https://localhost:8080/"} id="KFYg_qz8KUH8" outputId="01e0a0d8-d428-4897-cc69-cb726a05fe26"
print(lr.classes_)

# %% colab={"base_uri": "https://localhost:8080/"} id="CqoQNNjAKWQE" outputId="306286da-62b0-4214-baed-95423d37aac4"
print(lr.coef_.shape, lr.intercept_.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="6XonPnZPKcf0" outputId="229a6c72-bc42-4b2c-e193-2f4c9874729e"
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals = 2))

# %% colab={"base_uri": "https://localhost:8080/"} id="-FfYrAYcKjSE" outputId="1ee5f47c-7480-4343-eb0d-820ea86e1fd7"
from scipy.special import softmax
proba = softmax(decision, axis = 1)
print(np.round(proba, decimals = 3))

# %% id="HY_PXWEvKskk"
