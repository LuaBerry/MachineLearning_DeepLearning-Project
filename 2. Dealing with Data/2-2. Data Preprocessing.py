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

# %% id="50gu7NdQry6J"
import numpy as np

# %% id="mi_12aM5r55n"
np.column_stack(([1,2,3],[4,5,6]))

# %% id="5JvO1sR7r--r"
fish_data = np.column_stack((fish_length, fish_weight))

# %% colab={"base_uri": "https://localhost:8080/"} id="CrDflidKsJHb" outputId="361794c3-1777-432d-d664-ecac915a4d54"
print(fish_data[:5])

# %% colab={"base_uri": "https://localhost:8080/"} id="ZhBb_BRGsMJy" outputId="c3e10078-55ac-4b1f-e1df-d86f50600a11"
print(np.ones(5))

# %% id="U2qN-mrfsgAi"
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# %% colab={"base_uri": "https://localhost:8080/"} id="4h-W3zyisv77" outputId="1ab5479b-a03d-4bca-bf78-60a526fd7220"
print(fish_target)

# %% id="ACPSD2e-s2_i"
from sklearn.model_selection import train_test_split

# %% id="4QUuXH2etGAD"
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42)

# %% colab={"base_uri": "https://localhost:8080/"} id="PYocHbRytfB7" outputId="e62abaa4-9757-433a-e7be-ac5cf868607a"
print(train_input.shape, test_input.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="rcuF32jBuTtq" outputId="d9641fac-aeeb-4099-f19c-ef701e5136e6"
print(train_target.shape, test_target.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="IXub7U9nuYsb" outputId="7f646672-1983-497b-b456-75f896c2f487"
print(test_target)

# %% id="Ik8YbH6PueOb"
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42, stratify = fish_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="_I8cBDllutHT" outputId="25393280-5e80-4273-972e-f79032431e2e"
print(test_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="jAZBMr2-vItO" outputId="bf6ba0f3-9416-4227-892a-e6ed4efe3e00"
from sklearn.neighbors import KNeighborsClassifier as KNC
kn = KNC()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="zeEDqRG-vcVU" outputId="0b979fd4-aaf3-4a9f-fd8b-2102d55b5add"
print(kn.predict([[25,150]]))

# %% id="4ISoxvyGvilb"
import matplotlib.pyplot as plt
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% id="DrGZ4Gu5wKbH"
distances, indexes = kn.kneighbors([[25,150]])

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="zk1Ra-q7wczN" outputId="07ae849d-8459-457a-c3b2-d31ac868cb99"
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker = 'D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="mA8k6kUnw-MT" outputId="c41fc21d-5de3-4bb3-f223-679c79c43582"
print(train_input[indexes])

# %% colab={"base_uri": "https://localhost:8080/"} id="baJrpLafxBFy" outputId="7fb67b0d-b867-4f31-dbc9-4f34be90fd7e"
print(train_target[indexes])

# %% colab={"base_uri": "https://localhost:8080/"} id="Ugec2WEYxHns" outputId="e84638a2-65d5-48bf-ca57-0d08d7815dae"
print(distances)

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="FvKQWqmSxOUl" outputId="21bf40ae-c201-4416-adce-4a57816e9931"
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker = 'D')
plt.xlim((0,1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% id="HmVNueGexiTU"
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)

# %% colab={"base_uri": "https://localhost:8080/"} id="YWH6VSioy6Os" outputId="f69729e5-d389-4183-8e9e-460ab52b5be7"
print(mean, std)

# %% id="HYF6_11MzV_8"
train_scaled = (train_input - mean) / std

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="n5_4P1FBzcmj" outputId="e12956f0-38fa-4215-b342-41431862d01a"
plt.scatter(train_scaled[:, 0], train_scaled[:,1])
new = ([25, 150] - mean) / std
plt.scatter(new[0], new[1], marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% id="tsrwTYYR0AH0"
kn.fit(train_scaled, train_target)

# %% id="XU36UOJh0Gz7"
test_scaled = (test_input - mean) / std

# %% colab={"base_uri": "https://localhost:8080/"} id="yJI5pI6k0lQk" outputId="e3f57ed7-6403-4a88-ddeb-84f4b50c33f4"
kn.score(test_scaled, test_target)

# %% colab={"base_uri": "https://localhost:8080/"} id="goI3vBSN0n1j" outputId="7895307e-b971-4e88-e0b5-f91bcb7b89a0"
print(kn.predict([new]))

# %% colab={"base_uri": "https://localhost:8080/", "height": 279} id="bexZE9pT0rzD" outputId="cf3b82bb-341a-4a64-bdd4-7dd6bfcb71d6"
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker = 'D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# %% id="gTj6tAq-09uN"
  
