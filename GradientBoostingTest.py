import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('materials_data.csv')

features = ['density', 'debye_temperature', 'avg_atomic_mass', 'avg_electronegativity']
target = "youngs_modulus"

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

        # Initialize and train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01)
model.fit(X_train, y_train)

        # Make predictions
y_pred = model.predict(X_test)

        # Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE is {mse}")
print(f"r2 is {r2}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Young's Modulus")
plt.ylabel("Predicted Young's Modulus")
plt.show()
