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

# %% id="gMZXrcy7jlzC"
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# %% id="Jc-ps4wAj1Vo"
fish_data = [[l,w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1] * 35 + [0] * 14

# %% id="IObGXU4ykT3Y"
from sklearn.neighbors import KNeighborsClassifier as KNC
kn = KNC()

# %% id="NcyWZ5kQkhGh"
print(fish_data[:5])

# %% id="ZbzvVYGKks4l"
print(fish_data[44:])

# %% id="PdBfHdtpk4XS"
kn = kn.fit(fish_data[:35], fish_target[:35])
kn.score(fish_data[35:], fish_target[35:])

# %% id="HQFuCb13lYNA"
import numpy as np

# %% id="XJQH4sefmElR"
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# %% id="WL3bi1N0mOrA"
print(input_arr)

# %% id="C8V6kdzQmR-w"
print(input_arr.shape)

# %% id="Zopz_9GUmYt5"
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

# %% id="KGloGqiVnDBK"
print(index)

# %% id="At58yP1hnEaP"
print(input_arr[[1,3]])

# %% id="5UBgwEJYnR5h"
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

# %% colab={"base_uri": "https://localhost:8080/"} id="uYBy5LeBnkTs" outputId="cbe055f1-0015-43f8-9eb5-2250da03fe02"
print(input_arr[13], train_input[0])

# %% id="zAYvonm3ntj_"
test_input = input_arr[index[35:]]
test_target= target_arr[index[35:]]

# %% id="vwkMEgnun18B"
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% id="hfmeaMFIocZR"
kn = kn.fit(train_input, train_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="0JcS5x-HolQH" outputId="cf182744-356f-4cef-a160-04ed620dddb9"
kn.score(test_input, test_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="qRZv6Zx2ooSh" outputId="00440235-6e29-46c0-f2e9-f2b2763acf94"
kn.predict(test_input)

# %% colab={"base_uri": "https://localhost:8080/"} id="UUUqOWvzotCg" outputId="b8c1019e-ac12-4e1e-9910-9a31b5a1e170"
test_target

# %% id="rlYmxn6ooxBA"
