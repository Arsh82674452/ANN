# =========================================
# Customer Churn Prediction using ANN
# =========================================

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")

# Drop unnecessary columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# ===============================
# ENCODING
# ===============================
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# ===============================
# FEATURES & TARGET
# ===============================
X = df.drop(columns=["Exited"]).values
y = df["Exited"].values

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# BUILD MODEL
# ===============================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN MODEL (WITH EARLY STOPPING)
# ===============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# EVALUATE
# ===============================
y_pred = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {acc*100:.2f}%")

# ===============================
# FEATURE IMPORTANCE (FIXED)
# ===============================
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()[0]
        break

importance = np.mean(np.abs(weights), axis=1)

feature_names = df.drop(columns=["Exited"]).columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop Influencing Features:\n")
print(importance_df)

# ===============================
# SAVE MODEL + SCALER (FIXED)
# ===============================
os.makedirs("models", exist_ok=True)

model.save("models/model.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully in /models folder.")