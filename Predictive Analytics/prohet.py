import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error,
    mean_absolute_percentage_error, median_absolute_error, max_error,
    explained_variance_score, mean_pinball_loss, d2_tweedie_score, d2_pinball_score
)

# Load and robustly parse dates
df = pd.read_csv("train_cleansed.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True, errors='coerce')

# Aggregate monthly sales using 'ME' for month-end (future-proof)
monthly_data = df.resample('ME', on='Order Date').sum()
monthly_data.reset_index(inplace=True)
monthly_data = monthly_data.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

# Train/test split (last 12 months for testing)
data_train = monthly_data.iloc[:-12]
data_test = monthly_data.iloc[-12:]

# Fit Prophet model
model = Prophet(
    seasonality_mode='additive',
    seasonality_prior_scale=10,
    yearly_seasonality=True
)
model.fit(data_train)

# Make predictions
fitted_y = model.predict(data_train)
y_predict = model.predict(data_test)

actual_data = data_train['y']
y_pred1 = pd.Series(fitted_y['yhat'].values, index=data_train['ds'])
y_pred2 = pd.Series(y_predict['yhat'].values, index=data_test['ds'])

# Monthly sales for plotting (this aligns with previous approach)
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()

# Plot results
ax = monthly_sales.plot(style="-.", color="0.5", title="Predicting Using Prophet")
y_pred1.plot(ax=ax, linewidth=3, label="Prophet (Train)")
y_pred2.plot(ax=ax, linewidth=3, label="Prophet (Test)", color='C2')
plt.legend()
plt.show()

# Evaluation function
list_metrics = []
def evaluate_model(model_name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    me = max_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mpl = mean_pinball_loss(y_true, y_pred)
    d2t = d2_tweedie_score(y_true, y_pred)
    d2p = d2_pinball_score(y_true, y_pred)
    list_metrics.append([model_name, r2, mae, mse, msle, mape, medae, me, evs, mpl, d2t, d2p])
    print('{:^20} | {:^6} | {:^6} | {:^12} | {:^4} | {:^6} | {:^8} | {:^6} | {:^6} | {:^6} | {:^4} | {:^4} '.format(
         'Model','R2','MAE','MSE','MSLE','MAPE','MEDAE','ME','EVS','MPL','D2T','D2P'))
    formatl = ['{:<20} ','| {:^6.2f} ','| {:^6.0f} ','| {:^12,.0f}',' | {:^4.2f}',' | {:^6.2f}',' | {:^8.0f}',' | {:^6.0f}',' | {:^6.2f}',' | {:^6.0f}',
          ' | {:^4.2f} |',' {:^4.2f} ']
    for metrics in list_metrics:
        for format_, value in zip(formatl, metrics):
            print(format_.format(value), end="")
        print()

# Display R² scores and full metrics table
print('R² train:', r2_score(actual_data, y_pred1))
print('R² test:', r2_score(data_test['y'], y_pred2))
evaluate_model('Prophet', data_test['y'], y_pred2)
