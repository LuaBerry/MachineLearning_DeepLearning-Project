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

# %% id="jf8cCwpZArJ1"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state = 42)

# %% colab={"base_uri": "https://localhost:8080/"} id="3Joc3kfuBOhq" outputId="1ef72d38-f446-40aa-bd97-a0a641edda1a"
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(rf, train_input, train_target,
                        return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="1rrYdilIB-wa" outputId="905ba4f0-73c4-4309-c66d-b30900e81226"
rf.fit(train_input, train_target)
print(rf.feature_importances_)

# %% colab={"base_uri": "https://localhost:8080/"} id="jjG9uDXVCTkM" outputId="a29fb167-b3e7-4cd8-94a9-98099c3d9f5f"
rf = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 42)
rf.fit(train_input, train_target)
print(rf.oob_score_)

# %% colab={"base_uri": "https://localhost:8080/"} id="fYqfRa4ACvm8" outputId="804c9857-5ed7-4b36-fde1-9e86323cd723"
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(et, train_input, train_target,
                        return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="6Izbr-3dDnjf" outputId="36b221ce-b06a-4df5-feb1-a61b89e29b48"
et.fit(train_input, train_target)
print(et.feature_importances_)

# %% colab={"base_uri": "https://localhost:8080/"} id="L6qSsLpqEHlN" outputId="c85ad0c0-1c48-472f-a918-7b4681de8a68"
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state = 42)
scores = cross_validate(gb, train_input, train_target,
                        return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="1iAUHXrwEweS" outputId="2eab6a54-6260-4b2e-9fdd-5863e79abb09"
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.2, random_state = 42)
scores = cross_validate(gb, train_input, train_target,
                        return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="3qCI0XMZFAuV" outputId="b919b83e-25f6-4618-9876-7500b7449479"
gb.fit(train_input, train_target)
print(gb.feature_importances_)

# %% id="BTfl_hNpFK8d"
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state = 42)
scores = cross_validate(hgb, train_input, train_target,
                        return_train_score = True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="DTkuU09EGNkX" outputId="672007ac-a632-4a9e-c9f3-0e1cbb000721"
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target,
                                n_repeats = 10, random_state = 42, n_jobs = -1)
print(result.importances_mean)

# %% colab={"base_uri": "https://localhost:8080/"} id="6kgwroL1HO7l" outputId="ca8847fa-17a9-476d-9f85-589bc2617873"
result = permutation_importance(hgb, test_input, test_target,
                                n_repeats = 10, random_state = 42, n_jobs = -1)
print(result.importances_mean)

# %% colab={"base_uri": "https://localhost:8080/"} id="DPxLScCUHl29" outputId="16897726-c24c-4a88-a198-4d2710990d21"
hgb.score(test_input, test_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="7em5nF50Hqf9" outputId="3a862632-e878-419b-a8f8-0516d65cef73"
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method = 'hist', random_state = 42)
scores = cross_validate(xgb, train_input, train_target,
                        return_train_score = True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# %% colab={"base_uri": "https://localhost:8080/"} id="nq459kfHISZX" outputId="3d67a3c8-cf5b-42ce-a3d6-c597e890eb1e"
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state = 42)
scores = cross_validate(lgb, train_input,train_target,
                        return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
