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

# %% id="yMf-_ZbbGh7u"
import pandas as pd
fish = pd.read_csv("https://bit.ly/fish_csv_data")

# %% id="sp0uII36G18Y"
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# %% id="ooeT6zcdHNXr"
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

# %% id="CJqvsDb7HdEq"
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_scaled = ss.fit_transform(train_input)
test_scaled = ss.transform(test_input)

# %% colab={"base_uri": "https://localhost:8080/"} id="ArKLaQlcHuXk" outputId="e7dda897-dfb9-4aea-811b-429b8a9556c8"
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss = 'log', max_iter = 10, random_state = 42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="uinfztM4IKnD" outputId="88863e75-f342-4b70-945f-37c936947362"
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# %% id="ICPfSZM54Eko"
import numpy as np
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

# %% id="dXq-ok174f1u"
for _ in range(0, 300):
  sc.partial_fit(train_scaled, train_target, classes = classes)
  train_score.append(sc.score(train_scaled, train_target))
  test_score.append(sc.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="ueiGez-j40ob" outputId="14f20b3a-b60f-46c3-c2b5-6df38d9d3ced"
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="R5uXHwZ25Mxx" outputId="ab8f1441-1248-478a-f84c-9f9632faa468"
sc = SGDClassifier(loss='log', max_iter = 100, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="D7v0w_db5xDL" outputId="09dd94c9-30c1-45ad-b9bd-4ad860449f8f"
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="oG_3ej_y53hR" outputId="c07c7b3e-7a22-4504-afce-2664e8f8dc1d"
sc = SGDClassifier(loss = 'hinge', max_iter = 100, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
