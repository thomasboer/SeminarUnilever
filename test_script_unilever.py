import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
ts = pd.read_csv('aardappelpureedata.csv')

ts['date'] = pd.to_datetime(ts['date'])
ts.set_index('date', inplace=True)
ts.sort_index(inplace=True)
# period = len(ts)/2
print(ts.head())
decomposition = seasonal_decompose(ts,period=52)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print(auto_arima(ts['interest'],seasonal=True, m=52, start_p=0, start_q=0, max_P=1, max_Q=1).summary())

# plt.figure()
# plt.subplot(411)
# plt.plot(ts,label='Original')
# plt.subplot(412)
# plt.plot(seasonal,label='Seasonality')
# plt.subplot(413)
# plt.plot(trend,label='Trend')
# plt.subplot(414)
# plt.plot(residual,label='residuals')
# plt.show()


