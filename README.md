## DEVELOPED BY: S JAIGANESH
## REGISTER NO: 212222240037
## DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('raw_sales.csv')

# Convert 'datesold' to datetime and set as index
data['datesold'] = pd.to_datetime(data['datesold'])
data.set_index('datesold', inplace=True)

# Plot the time series data
plt.plot(data.index, data['price'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Time Series')
plt.show()

# Function to check stationarity using ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['price'])

# Plot ACF and PACF to determine SARIMA parameters
plot_acf(data['price'])
plt.show()
plot_pacf(data['price'])
plt.show()

# Train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['price'][:train_size], data['price'][train_size:]

# Define and fit the SARIMA model on training data
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions on the test set
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot the actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/52346d8e-0844-44f3-b936-fed1d716e1e2)
![image](https://github.com/user-attachments/assets/dc17f772-0e47-419d-b507-bffe3e266f4e)
![image](https://github.com/user-attachments/assets/c57e30d8-f75d-4bdb-8c21-11ee1dde7b1b)
![image](https://github.com/user-attachments/assets/44305b0b-6083-41fb-8d7b-97e598c8704e)
![image](https://github.com/user-attachments/assets/77751eba-d7b0-4ca6-aac4-9352a0744769)


### RESULT:
Thus, the pyhton program based on the SARIMA model is executed successfully.
