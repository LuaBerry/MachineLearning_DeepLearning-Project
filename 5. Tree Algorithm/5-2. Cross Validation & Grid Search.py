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

# %% id="0lxtqUq2gpjq"
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# %% id="lNoGB2__gxoI"
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# %% id="Fa9OgWogg8_w"
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state = 42, test_size = 0.2)
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, random_state = 42, test_size = 0.2)

# %% colab={"base_uri": "https://localhost:8080/"} id="YfiHrV1OhuG5" outputId="b5b4467e-3f0e-43a2-895d-eb267c0cb827"
print(sub_input.shape, val_input.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="9fKxOY45iQ63" outputId="c4c4f8f0-eb9f-4582-c1d8-934c68e74b6a"
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(sub_input,sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="y-PwCl02ijaI" outputId="05742686-0899-430a-bb8a-fa9745fc8b16"
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)

# %% colab={"base_uri": "https://localhost:8080/"} id="4lXBB3ecjh4o" outputId="0cf4c7fe-a958-49b0-c746-2b9e8387e499"
import numpy as np
print(np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="uyI0qWhHjmv3" outputId="9b9ef12c-765b-4c99-fd21-2569ea5a0a01"
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
print(np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="RQiC-axckEgw" outputId="ec42e021-45f4-4e9f-996d-5350f60e6854"
splitter = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
scores = cross_validate(dt, train_input, train_target, cv = splitter)
print(np.mean(scores['test_score']))

# %% id="FF42R_GWkU5p"
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

# %% id="CiLW_ZZjltRR"
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)

# %% id="ib_RJT1jl3oY"
gs.fit(train_input, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="iG4-Ws87mA-4" outputId="ada9341e-853e-4912-b8c0-fe2a3c7bf510"
dt = gs.best_estimator_
print(dt.score(train_input, train_target))

# %% colab={"base_uri": "https://localhost:8080/"} id="J4G62FD4mMS4" outputId="7a59cb02-35ac-45a4-b782-891dcf94b9fa"
print(gs.best_params_)

# %% colab={"base_uri": "https://localhost:8080/"} id="WuJ4XSDNmiHA" outputId="4fda5b30-c40e-41d5-ab22-ab21bbef6893"
print(gs.cv_results_['mean_test_score'])

# %% colab={"base_uri": "https://localhost:8080/"} id="Ls0kanazmnih" outputId="e1452080-bb76-4241-b345-2b47d95c3b6e"
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

# %% id="ON-PEbKfnLII"
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
          'max_depth' : range(5, 20, 1),
          'min_samples_split' : range(2, 22, 2)}

# %% colab={"base_uri": "https://localhost:8080/"} id="Qb46mw42nvfh" outputId="b734b423-a4f5-407c-dd37-6a6172e00ddc"
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
gs.fit(train_input, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="Hm8xkjbZoA3w" outputId="d81b20be-39e4-4644-f01b-ec7bb783989f"
print(gs.best_params_)

# %% colab={"base_uri": "https://localhost:8080/"} id="sHjQMYWIoNrQ" outputId="eb47ff7d-60bb-4402-e8fc-a8727bb39bd0"
print(np.max(gs.cv_results_['mean_test_score']))

# %% id="wLR4LMDHoe5Z"
from scipy.stats import uniform, randint

# %% colab={"base_uri": "https://localhost:8080/"} id="szYqONTRqZAV" outputId="a09357f4-cdbd-4620-8372-9a6c50ff872b"
rgen = randint(0, 10)
rgen.rvs(10)

# %% colab={"base_uri": "https://localhost:8080/"} id="wu1Q2KGhqcNL" outputId="089dabb3-3924-4948-f3e6-52c32d551ae4"
np.unique(rgen.rvs(1000), return_counts = True)

# %% colab={"base_uri": "https://localhost:8080/"} id="NeUKCJRgqkAc" outputId="648239c6-6ade-4012-f476-3f38232fcd89"
ugen = uniform(0, 1)
ugen.rvs(10)

# %% id="hAOXmYHrqqy6"
params = {'min_impurity_decrease' : uniform(0.0001, 0.001),
          'max_depth' : randint(20, 50),
          'min_samples_split' : randint(2, 25),
          'min_samples_leaf' : randint(1, 25),
          }

# %% id="vUjVAJi6rSFC"
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42), params,
                        n_iter = 100, n_jobs = -1, random_state = 42)
gs.fit(train_input, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="oVX_SPUgrmU7" outputId="7f4507d6-2894-4a02-c9cd-3ded856a8b49"
print(gs.best_params_)

# %% colab={"base_uri": "https://localhost:8080/"} id="pm3zsOP1rwCa" outputId="b9226e6c-e61e-4826-e6a4-e3983f5d8d39"
print(np.max(gs.cv_results_['mean_test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="Jm6vt1Cur4ti" outputId="99c6fa9c-9b0b-4ccc-d0ca-4d83a767dc29"
dt = gs.best_estimator_
print(dt.score(test_input, test_target))

# %% id="b_vmR1QLr-Kj"
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42, splitter = 'random'), params,
                        n_iter = 100, n_jobs = -1, random_state = 42)
gs.fit(train_input, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="56pqfPuztlLb" outputId="8b719bd3-4593-41fb-8ed9-aa9ed99873fd"
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
