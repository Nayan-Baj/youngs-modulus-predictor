import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
df = pd.read_csv("materials_data.csv")
y = df["youngs_modulus"]

# -------- Model 1 --------
X1 = df[['density', 'debye_temperature', 'band_gap', 'avg_atomic_mass', 'avg_atomic_radius']]
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X1, y)
joblib.dump(model1, "model_1.pkl")
print("Saved model_1.pkl")

# -------- Model 2 --------
X2 = df[['density', 'debye_temperature', 'avg_atomic_mass', 'avg_atomic_radius']]
model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X2, y)
joblib.dump(model2, "model_2.pkl")
print("Saved model_2.pkl")

# -------- Model 3 --------
X3 = df[['debye_temperature', 'volume', 'avg_atomic_mass']]
model3 = RandomForestRegressor(n_estimators=100, random_state=42)
model3.fit(X3, y)
joblib.dump(model3, "model_3.pkl")
print("Saved model_3.pkl")

# -------- Model 5 --------
X4 = df[['debye_temperature', 'avg_atomic_mass', 'avg_electronegativity']]
model4 = RandomForestRegressor(n_estimators=100, random_state=42)
model4.fit(X4, y)
joblib.dump(model4, "model_4.pkl")
print("Saved model_4.pkl")
