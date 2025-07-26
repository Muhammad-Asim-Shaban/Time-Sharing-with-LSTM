# ⏱️ Time Series Forecasting with LSTM 🧠📈
This project demonstrates how to build a Long Short-Term Memory (LSTM) neural network to forecast time series data — in this case, the monthly total number of airline passengers 📊.

# 📁 Dataset
The dataset used is:

🗃️ air-passenger.csv
📌 Column: #Passengers
📅 Monthly data (1949–1960)
🔢 Goal: Predict future number of passengers

# 🧰 Libraries Used
NumPy & Pandas – Data manipulation

Matplotlib – Plotting 📉

Keras – LSTM model building

Sklearn – MinMaxScaler & RMSE

Warnings – To suppress logs

# 🔄 Data Preprocessing
Converted passenger counts to float32

Normalized data using MinMaxScaler

Split into training (67%) and testing (33%)

```python
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset.reshape(-1, 1))
Created sequences using a sliding window (look_back=1):
```
```python
def createDataset(dataset, look_back=1):
    datax, datay = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), 0]
        datax.append(a)
        datay.append(dataset[i + look_back, 0])
    return np.array(datax), np.array(datay)
```
# 🧠 Model Architecture (LSTM)
1 LSTM layer

1 Dense layer for output

Compiled with mean_squared_error loss and adam optimizer

```python
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```
🧪 Trained for 100 epochs with batch size = 1

# 📈 Evaluation & Visualization
Evaluated using Root Mean Squared Error (RMSE) on training and test data

Plotted:

Actual vs predicted values

Predictions for training and testing data

```python
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
```
