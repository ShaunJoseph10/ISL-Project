import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "data"

X = []
y = []

print("Reading dataset...")

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for file_name in os.listdir(label_path):

        if file_name.endswith(".npy"):
            file_path = os.path.join(label_path, file_name)

            data = np.load(file_path)
            data = data.flatten()

            # 🔥 Accept 42-length data
            if len(data) == 42:
                X.append(data)
                y.append(label)

print(f"Total samples collected: {len(X)}")

if len(X) == 0:
    print("❌ No valid training data found.")
    exit()

print("Training model...")

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

with open("isl_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as isl_model.pkl")
