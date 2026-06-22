import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor

from utils import build_interval_dataset

# Load configuration

CONFIG_PATH = "configs/stress.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

df = pd.read_parquet(config["data_path"])

features = config["features"]
target = config["target"]

# Build the dataset with interval features and target

df = build_interval_dataset(df, features, target)

# Split the data into training and testing sets

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Train the LightGBM model

model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)

mae = mean_absolute_error(y_test, pred)

mae_percent = mae / y_test.mean() * 100

print()
print("===== LightGBM =====")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAE (%): {mae_percent:.2f}%")