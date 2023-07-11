import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Step 1: Data Collection
# Assuming you have historical Bitcoin price data in a CSV file with columns: Date, Open, High, Low, Close
data = pd.read_csv('bitcoin_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-6226461823254193"
     crossorigin="anonymous"></script>
# Step 2: Feature Engineering
# Assuming you want to use the previous day's prices as features
data['Prev_Open'] = data['Open'].shift(1)
data['Prev_High'] = data['High'].shift(1)
data['Prev_Low'] = data['Low'].shift(1)
data['Prev_Close'] = data['Close'].shift(1)

# Step 3: Split Data into Train and Test
X = data.dropna().drop(['Date', 'Close'], axis=1)
y = data.dropna()['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_train_pred = model.predict(X_train)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
y_test_pred = model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Step 6: Future Price Prediction
# Assuming you want to predict the next day's price
latest_data = data.iloc[-1].drop(['Date', 'Close'])
latest_data = latest_data.values.reshape(1, -1)
next_day_price = model.predict(latest_data)

# Print the predicted price and evaluation metrics
print(f"Next Day's Predicted Price: {next_day_price[0]:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
