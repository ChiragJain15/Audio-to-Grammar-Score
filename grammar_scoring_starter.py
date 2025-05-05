# ==========================
# 📌 1. Imports and Setup
# ==========================
import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================
# 📌 2. Load Data
# ==========================
train_df = pd.read_csv('./Dataset/train.csv')
test_df = pd.read_csv('./Dataset/test.csv')

# ==========================
# 📌 3. Load Pretrained Wav2Vec2
# ==========================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
model.eval()

# ==========================
# 📌 4. Feature Extraction
# ==========================
def extract_features(file_path):
    waveform, sr = torchaudio.load(file_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        embeddings = model(input_values).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

# ==========================
# 📌 5. Extract Features for Train
# ==========================
X, y = [], []

# First pass to get feature dimension
sample_file = os.path.join("Dataset/audios/train", train_df.iloc[0]["filename"])
sample_feat = extract_features(sample_file)
feature_dim = sample_feat.shape[0]

for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    file_path = os.path.join("Dataset/audios/train", row["filename"])
    try:
        feat = extract_features(file_path)
        # Ensure feature vector has correct dimension
        if feat.shape[0] == feature_dim:
            X.append(feat)
            y.append(row["label"])
        else:
            print(f"Warning: Skipping {row['filename']} due to incorrect feature dimension")
    except Exception as e:
        print(f"Error processing {row['filename']}: {str(e)}")

X = np.array(X)
y = np.array(y)

# ==========================
# 📌 6. Train-Validation Split
# ==========================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
y_pred = model_ridge.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"📉 Validation RMSE: {rmse:.4f}")

# ==========================
# 📌 7. Retrain on Full Data
# ==========================
model_ridge.fit(X, y)

# ==========================
# 📌 8. Predict on Test Data
# ==========================
X_test = []
test_filenames = []

for fname in tqdm(test_df["filename"]):
    file_path = os.path.join("Dataset/audios/test", fname)
    try:
        feat = extract_features(file_path)
        # Ensure feature vector has correct dimension
        if feat.shape[0] == feature_dim:
            X_test.append(feat)
            test_filenames.append(fname)
        else:
            print(f"Warning: Skipping {fname} due to incorrect feature dimension")
    except Exception as e:
        print(f"Error processing {fname}: {str(e)}")

X_test = np.array(X_test)
test_preds = model_ridge.predict(X_test)

submission = pd.DataFrame({"filename": test_filenames, "label": test_preds})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")

# ==========================
# 📌 9. Final Notes
# ==========================
print(f"✅ Final RMSE on train split: {rmse:.4f}")

# ==========================
# 📈 Optional Visualization
# ==========================
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.7)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("Validation: True vs Predicted Grammar Scores")
plt.grid()
plt.show()
