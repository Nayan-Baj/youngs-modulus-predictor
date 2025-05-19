import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("materials_data.csv")

# List of all possible features
all_features = ["density", "debye_temperature", "volume", "band_gap",
                "avg_coordination_number", "avg_atomic_mass",
                "avg_electronegativity", "avg_atomic_radius"]

# Target
y = df["youngs_modulus"]

# To store results
results = []

# Try every non-empty combination of features
for r in range(1, len(all_features) + 1):
    print(r)
    for feature_combo in itertools.combinations(all_features, r):
        feature_list = list(feature_combo)

        X = df[feature_list]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        results.append((r2, feature_list))

# Sort results by R² descending
results.sort(reverse=True, key=lambda x: x[0])

# Print best feature sets
print("Top feature combinations by R²:")
for r2, features in results:
    print(f"R²: {r2:.4f} - Features: {features}")

# If you want, retrain model on the best set
best_features = results[0][1]
print(f"\nBest features: {best_features}")

X_best = df[best_features]
X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Final plot
r2_final = r2_score(y_test, y_pred)
plt.scatter(y_test, y_pred)
plt.xlabel("True Young's Modulus (GPa)")
plt.ylabel("Predicted Young's Modulus (GPa)")
plt.title(f"Best Random Forest Model\nR² = {r2_final:.4f}")
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.show()
