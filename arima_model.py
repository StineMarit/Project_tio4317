import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

from historical_data import get_historical_data

# Set parameters
rate = 'EUR'
training_start = pd.Timestamp('2018-01-01')
training_end = pd.Timestamp('2023-12-31')
prediction_end = pd.Timestamp('2024-12-31')

output_file = open(f'results\\result.txt', 'w')

# Collect input data
df = get_historical_data(start_date=training_start, end_date=prediction_end, rate=rate)
df.to_csv("data\\input_data.csv")
training_df = df.loc[:training_end]
test_df = df.loc[training_end:]


# Perform Differencing to estimate d
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 4), sharex=True)
ax1.plot(training_df['exchange_rate'].diff(), color='orange')
ax1.set_title('1st Order Differencing')
ax1.axes.xaxis.set_visible(False)
ax2.plot(training_df['exchange_rate'].diff().diff(), color='green')
ax2.set_title('2nd Order Differencing')
plt.tight_layout()
plt.savefig(f"results\\differencing.png")
plt.show()
d = 1

# Estimate Best Values for p and q based on AIC, BIC, and HQIC
lags = 12
diff_series = training_df['exchange_rate'].diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
plot_acf(diff_series, lags=lags, ax=axes[0])
axes[0].set_title('ACF of First Order Differenced Series')
plot_pacf(diff_series, lags=lags, ax=axes[1])
axes[1].set_title('PACF of First Order Differenced Series')
plt.tight_layout()
plt.savefig(f"results\\acp_pacf.png")
plt.show()

order = 3
results = []
for p in range(order):
    for q in range(order):
        model = ARIMA(training_df['exchange_rate'], exog=training_df.drop(columns=['exchange_rate']), order=(p, d, q))
        result = model.fit()
        results.append({
               'p': p,
               'd': d,
               'q': q,
               'AIC': result.aic,
               'BIC': result.bic,
               'HQIC': result.hqic
           })
results_df = pd.DataFrame(results)
output_file.write("Estimate Best Values for p and q\n")
output_file.write(str(results_df))
output_file.write(f"\nBest model based on AIC:\n{results_df.loc[results_df['AIC'].idxmin()]}")
output_file.write(f"\nBest model based on BIC:\n{results_df.loc[results_df['BIC'].idxmin()]}")
output_file.write(f"\nBest model based on HQIC:\n{results_df.loc[results_df['HQIC'].idxmin()]}")
p, q = 2, 0

# Fit an ARIMA model
model = ARIMA(training_df['exchange_rate'], exog=training_df.drop(columns=['exchange_rate']), order=(p, d, q))
model_fit = model.fit()
output_file.write('\n'+str(model_fit.summary()))

# Ljung-Box test
residuals = model_fit.resid
ljung_box_results = acorr_ljungbox(residuals, lags=[12], return_df=True)
output_file.write("\nLjung-Box Test Results:")
output_file.write(str(ljung_box_results))

# Augmented Dickey-Fuller test
adf_stat, p_value, _, _, critical_values, _ = adfuller(residuals)
output_file.write("\n\nAugmented Dickey-Fuller Test:")
output_file.write(f"\nADF Statistic: {adf_stat}")
output_file.write(f"\nP-value: {p_value}")
output_file.write(f"\nCritical Values: {critical_values}")

# Shapiro-Wilk Test
shapiro_stat, shapiro_p_value = stats.shapiro(residuals)
output_file.write("\n\nShapiro-Wilk Test:")
output_file.write(f"\nStatistic: {shapiro_stat}")
output_file.write(f"\nP-value: {shapiro_p_value}")

# 4. Breusch-Pagan Test
# bp_test = het_breuschpagan(residuals, model_fit.model.exog)
# bp_results = pd.DataFrame({
#     'Lagrange Multiplier Statistic': bp_test[0],
#     'p-value': bp_test[1],
#     'f-value': bp_test[2],
#     'f-test p-value': bp_test[3]
# }, index=['Breusch-Pagan Test'])
# output_file.write("\n\nBreusch-Pagan Test Results:")
# output_file.write('\n' + str(bp_results))

# Diagnostic plots
model_fit.plot_diagnostics(figsize=(12, 8))
plt.savefig(f"results\\diagnostics.png")
plt.show()

# Forecasting
forecast_index = test_df.index
forecast = model_fit.get_forecast(steps=len(forecast_index), exog=test_df.drop(columns=['exchange_rate']))
forecast_df = pd.DataFrame({'forecast': forecast.predicted_mean}, index=forecast_index)

# Confidence intervals
confidence_intervals = [95, 80, 65, 50]
for ci in confidence_intervals:
    data = forecast.conf_int(alpha=(100-ci)/100)
    forecast_df[f'lower_bound_{ci}'] = data.iloc[:, 0]
    forecast_df[f'upper_bound_{ci}'] = data.iloc[:, 1]

forecast_df.to_csv("data\\forecast.csv")

# Combine actual and forecast for visualization
plt.figure(figsize=(12, 6))
plt.plot(df['exchange_rate'], label='Actual Exchange Rate', color='blue')
plt.plot(forecast_df['forecast'], label='Forecasted Exchange Rate', color='orange', linestyle='--')
plt.title('Actual Exchange Rate and Forecast using ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.savefig(f"results\\actual_forecast.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df.loc[training_end:, 'exchange_rate'], label='Actual', color='blue')
plt.plot(forecast_df['forecast'], label='Forecast', color='orange')
plt.title('Forecast Result')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(f"results\\forecast.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df.loc[training_end:, 'exchange_rate'], label='Actual', color='blue')
plt.plot(forecast_df['forecast'], label='Forecast', color='orange')
start_transparency = 0.3
for ci in confidence_intervals:
    plt.fill_between(forecast_df.index,
                     forecast_df[f'lower_bound_{ci}'],
                     forecast_df[f'upper_bound_{ci}'],
                     color='grey', alpha=start_transparency, label=f'{ci}% Confidence Interval')
    start_transparency += 0.2
plt.title('Forecast Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(f"results\\ci.png")
plt.show()

output_file.close()
