import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from historical_data import get_historical_data


"""
Comments:
    - Highest log likelihood so far 23.806 with p=3, d=1, q=1 (eur_2018_2023_2024_p3_d1_q1_result.txt)
"""


# Set parameters
rate = 'EUR'
training_start = pd.Timestamp('2018-01-01')
training_end = pd.Timestamp('2023-12-31')
prediction_end = pd.Timestamp('2024-12-31')
p, d, q = 3, 1, 1  # Todo: set  # Define the order of the ARIMA model; (p, d, q)
run_name = f'{rate}_{training_start.year}_{training_end.year}_{prediction_end.year}_p{p}_d{d}_q{q}'.lower()

# Collect input data
df = get_historical_data(start_date=training_start, end_date=prediction_end, rate=rate)
training_df = df.loc[:training_end]
test_df = df.loc[training_end:]

# Fit an ARIMA model
model = ARIMA(training_df['exchange_rate'], exog=training_df.drop(columns=['exchange_rate']), order=(p, d, q))

# Fit the model
model_fit = model.fit()

# Output model summary
print(model_fit.summary())
try:
    f = open(f"test_figures_and_results\\{run_name}_result.txt", "x")
except FileExistsError:
    f = open(f"test_figures_and_results\\{run_name}_result.txt", "w")
f.write(str(model_fit.summary()))

# Diagnostic plots
model_fit.plot_diagnostics(figsize=(12, 8))
plt.savefig(f"test_figures_and_results\\{run_name}_diagnostics.png")
plt.show()

# Step 4: Forecasting
forecast_index = test_df.index
forecast = model_fit.forecast(steps=len(forecast_index), exog=test_df.drop(columns=['exchange_rate']))
forecast_df = pd.DataFrame(list(forecast), columns=['Forecast'], index=forecast_index)

# Step 5: Combine actual and forecast for visualization
plt.figure(figsize=(12, 6))
plt.plot(df['exchange_rate'], label='Actual Exchange Rate', color='blue')
plt.plot(forecast_df, label='Forecasted Exchange Rate', color='orange', linestyle='--')
plt.title('Exchange Rate Forecast using ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.savefig(f"test_figures_and_results\\{run_name}_forecast.png")
plt.show()

