import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Data Frame on which the model is trained

data = pd.read_csv('ML major project/train.csv')
print('Data shape before cleaning:', data.shape) # to check the shape of the data
data.isnull().sum()
data.duplicated().sum()

# Cleaning the data
data = data.dropna()
data = data.drop_duplicates()
print('Data shape after cleaning:', data.shape)

# splitting the data into features and target variable
features = [
    'ambient_temperature',
    'generator_speed',
    'generator_winding_temp_max',
    'nc1_inside_temp',
    'nacelle_temp',
    'wind_direction_raw',
    'wind_speed_raw',
    'wind_speed_turbulence'
]
X = data[features]
y = data['Target']

# normalizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# creating the models
model = RandomForestRegressor(n_estimators=100, random_state=42)

# training the model
model.fit(X_train, y_train)

# predicting the target variable
y_pred = model.predict(X_test)

# evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# visualization 
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Target Values")
plt.show()