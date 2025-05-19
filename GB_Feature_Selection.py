import itertools
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("materials_data.csv")

features = ["density", "debye_temperature", "volume", "band_gap",
                "avg_coordination_number", "avg_atomic_mass",
                "avg_electronegativity", "avg_atomic_radius"]
target = "youngs_modulus"

best_features = None
#best_mse = float('inf')
best_r2 = float('-inf')

# Iterate through all possible feature combinations
print(len(features))
for r in range(1, len(features) + 1):
    print(f"testing + {r}")
    for combo in itertools.combinations(features, r):
        X_train, X_test, y_train, y_test = train_test_split(df[list(combo)], df[target], test_size=0.2, random_state=42)

        # Initialize and train XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.05)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        r2 = r2_score(y_test, y_pred)

        # Check if it's the best combination so far
        if r2 > best_r2:
            best_r2 = r2
            best_features = combo
            best_y_test = y_test
            best_y_pred = y_pred

# Print the best combination of features
print(f'Best feature combination: {best_features} with r2: {best_r2}')

# Plot actual vs predicted values for the best model
plt.scatter(best_y_test, best_y_pred, alpha=0.5)
plt.xlabel("Actual Young's Modulus")
plt.ylabel("Predicted Young's Modulus")
plt.title(f"Best Model - Features: {best_features}")
plt.show()
