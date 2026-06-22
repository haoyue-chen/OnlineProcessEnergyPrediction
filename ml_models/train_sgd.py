import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from utils import build_interval_dataset


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

CONFIG_PATH = "configs/stress.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

df = pd.read_parquet(config["data_path"])

features = config["features"]
target = config["target"]


# --------------------------------------------------
# Build interval-level dataset
# --------------------------------------------------

df = build_interval_dataset(
    df,
    features,
    target
)

print(f"Intervals: {len(df)}")


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# SGD Regressor
# --------------------------------------------------

model = SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=0.0001,
    max_iter=5000,
    tol=1e-4,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

r2 = r2_score(y_test, pred)

mae = mean_absolute_error(y_test, pred)

mae_percent = mae / y_test.mean() * 100

print()
print("===== SGD Regressor =====")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAE (%): {mae_percent:.2f}%")