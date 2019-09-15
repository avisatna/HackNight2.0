from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

model = load_model('model.h5')
sc = MinMaxScaler(feature_range = (0, 1))
dataset = pd.read_csv('005930.KS.csv', index_col="Date", parse_dates=True)

dataset_test = pd.read_csv('005930.KS_test.csv', index_col="Date",parse_dates=True)
dataset_test = dataset_test.dropna()
real_stock_price = dataset_test.iloc[:, 3:4].values
dataset_test["Volume"] = dataset_test["Volume"].replace(',', '').astype(float)
test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)

dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)
X_test = []
for i in range(60, 60+len(dataset_test)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price[0])