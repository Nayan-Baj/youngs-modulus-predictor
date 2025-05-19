import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Loading dataset
df = pd.read_csv("materials_data.csv")  # assuming your file is saved

# Preparing X and y
X = df[["avg_atomic_mass", "debye_temperature", "avg_atomic_radius", "band_gap"]]
y = df["youngs_modulus"]  # target: young's modulus

# Test/Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# R^2 Value
r2 = r2_score(y_test, y_pred)
print(f"R² score: {r2:.4f}")
print(f"Max Young's modulus in dataset: {y.max()}")
print(f"Min Young's modulus in dataset: {y.min()}")

plt.scatter(y_test, y_pred)
plt.xlabel("True Young's Modulus (GPa)")
plt.ylabel("Predicted Young's Modulus (GPa)")
plt.title(f"Random Forest Regression\nR² = {r2:.4f}")
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")  # perfect prediction line
plt.show()
