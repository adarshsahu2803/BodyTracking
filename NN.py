import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

train_data = pd.read_csv("WEC/train.csv")

trainx = train_data[['F1','F2','F4','F5','F6','F7','F8','F10','F11']]
trainY = train_data[['Expected']]
train_index = train_data[['Id']]

test_data = pd.read_csv("WEC/test.csv")

testx = test_data[['F1','F2','F4','F5','F6','F7','F8','F10','F11']]
# testY = pd.read_csv("WEC/output.csv")
# testY = testY[['Expected']]
test_index = test_data[['Id']]

regressor = LogisticRegression()
regressor.fit(trainx, trainY)

# print(test_index)

y_pred = regressor.predict(testx)

test_index = test_index.to_numpy(dtype=int)

dataout = np.concatenate((test_index.reshape(len(test_index),1),y_pred.reshape(len(y_pred),1)),1)
# print(dataout)
datacsv = pd.DataFrame(dataout, columns=['Id','Expected'])
datacsv['Id'] = datacsv['Id'].astype(int)
# print(datacsv)
datacsv.to_csv("Predicted.csv", index=False)


# print(trainx)
# print(trainY)

# from sklearn.preprocessing import StandardScaler
# sctrain = StandardScaler()
# X_train = sctrain.fit_transform(trainx)
# X_test = sctrain.transform(testx)

# sctest = StandardScaler()
# X_train = sctrain.fit_transform(trainx)
# X_test = sctrain.transform(testx)

# trainx.to_numpy()
# trainY.to_numpy()

# model = keras.models.Sequential()
# model.add(keras.layers.Dense(9, input_dim = 9, activation = 'relu'))
# model.add(keras.layers.Dense(15, activation = 'relu'))
# model.add(keras.layers.Dense(1, activation = 'relu'))

# adam = Adam(0.001)
# model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

# print(model.summary())

# history = model.fit(trainx, trainY, epochs=50, verbose=1, batch_size=20)

# print(history.history)
