# tennis_winner_prediction_full.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os

# -------------------------------
# 1. Load data
# -------------------------------
csv_path = "atp_matches_2024.csv"  # change path if needed
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

# -------------------------------
# 2. Feature engineering
# -------------------------------
def safe_div(a, b):
    """Avoid division by zero."""
    return np.where((b == 0) | (pd.isna(b)), np.nan, a / b)

df['rank_diff'] = df['loser_rank'] - df['winner_rank']
df['age_diff'] = df['winner_age'] - df['loser_age']
df['ht_diff'] = df['winner_ht'] - df['loser_ht']
df['ace_diff'] = df['w_ace'] - df['l_ace']
df['df_diff'] = df['l_df'] - df['w_df']
df['w_bp_ratio'] = safe_div(df['w_bpSaved'], df['w_bpFaced'])
df['l_bp_ratio'] = safe_div(df['l_bpSaved'], df['l_bpFaced'])
df['bp_save_diff'] = df['w_bp_ratio'] - df['l_bp_ratio']
df['rank_points_diff'] = df['winner_rank_points'] - df['loser_rank_points']

important_cols = [
    'rank_diff', 'age_diff', 'ht_diff', 'ace_diff', 
    'df_diff', 'bp_save_diff', 'rank_points_diff'
]

before_drop = len(df)
df = df.dropna(subset=important_cols)
after_drop = len(df)
print(f"Dropped {before_drop - after_drop} rows with missing feature data.")
if len(df) == 0:
    raise ValueError("❌ No valid rows remain after dropping NaNs. Check your CSV data!")

# -------------------------------
# 3. Prepare features and target
# -------------------------------
X = df[important_cols].values
y = np.ones(len(X))

df_flipped = df.copy()
for f in important_cols:
    df_flipped[f] = -df_flipped[f]
X_flipped = df_flipped[important_cols].values
y_flipped = np.zeros(len(X_flipped))

X_all = np.vstack([X, X_flipped])
y_all = np.concatenate([y, y_flipped])
print(f"Final dataset size after flipping: {X_all.shape[0]} samples")

# -------------------------------
# 4. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# -------------------------------
# 5. Normalize features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Logistic Regression (Gradient Descent)
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, lr=0.05, epochs=10000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for i in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        w -= lr * dw
        b -= lr * db
        if i % 2000 == 0:
            loss = -(1/m)*np.sum(y*np.log(y_hat+1e-9)+(1-y)*np.log(1-y_hat+1e-9))
            print(f"Epoch {i}: Loss = {loss:.4f}")
    return w, b

print("Training logistic regression using gradient descent...")
w, b = logistic_regression_gd(X_train_scaled, y_train, lr=0.05, epochs=10000)

# -------------------------------
# 7. Evaluate model
# -------------------------------
def evaluate(X, y, w, b, dataset_name):
    probs = sigmoid(np.dot(X, w) + b)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    print(f"\n{dataset_name} Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC:  {auc:.3f}")
    return probs

train_probs = evaluate(X_train_scaled, y_train, w, b, "Training")
test_probs = evaluate(X_test_scaled, y_test, w, b, "Test")

# -------------------------------
# 8. Feature importance
# -------------------------------
coef_df = pd.DataFrame({
    'feature': important_cols,
    'weight': w
}).sort_values(by='weight', ascending=False)
print("\nFeature Importance (Weights):")
print(coef_df.to_string(index=False))

# -------------------------------
# 9. Build player stats for name-based prediction
# -------------------------------
def build_player_stats(df):
    player_stats = {}
    winner_cols = {'name':'winner_name','rank':'winner_rank','age':'winner_age',
                   'ht':'winner_ht','ace':'w_ace','df':'w_df',
                   'bpSaved':'w_bpSaved','bpFaced':'w_bpFaced','rank_points':'winner_rank_points'}
    loser_cols  = {'name':'loser_name','rank':'loser_rank','age':'loser_age',
                   'ht':'loser_ht','ace':'l_ace','df':'l_df',
                   'bpSaved':'l_bpSaved','bpFaced':'l_bpFaced','rank_points':'loser_rank_points'}
    w = df[list(winner_cols.values())].copy()
    w.columns = winner_cols.keys()
    l = df[list(loser_cols.values())].copy()
    l.columns = loser_cols.keys()
    all_players = pd.concat([w, l], ignore_index=True)
    grouped = all_players.groupby('name').agg({
        'rank':'mean','age':'mean','ht':'mean','ace':'mean','df':'mean',
        'bpSaved':'mean','bpFaced':'mean','rank_points':'mean'
    })
    grouped['bp_ratio'] = grouped['bpSaved'] / grouped['bpFaced']
    grouped = grouped.dropna()
    return grouped

player_stats = build_player_stats(df)
print(f"✅ Built stats for {len(player_stats)} players")

# -------------------------------
# 10. Prediction function
# -------------------------------
def predict_match(player1_name, player2_name, player_stats, scaler, w, b):
    if player1_name not in player_stats.index or player2_name not in player_stats.index:
        print("❌ One or both players not found in dataset.")
        return None
    p1 = player_stats.loc[player1_name]
    p2 = player_stats.loc[player2_name]

    rank_diff = p2['rank'] - p1['rank']
    age_diff = p1['age'] - p2['age']
    ht_diff = p1['ht'] - p2['ht']
    ace_diff = p1['ace'] - p2['ace']
    df_diff = p2['df'] - p1['df']
    bp_save_diff = p1['bp_ratio'] - p2['bp_ratio']
    rank_points_diff = p1['rank_points'] - p2['rank_points']

    X_new = np.array([[rank_diff, age_diff, ht_diff, ace_diff, df_diff, bp_save_diff, rank_points_diff]])
    X_new_scaled = scaler.transform(X_new)
    prob = sigmoid(np.dot(X_new_scaled, w) + b)[0]

    print(f"\nPrediction: {player1_name} vs {player2_name}")
    print(f"→ Probability {player1_name} wins: {prob*100:.2f}%")
    print(f"→ Probability {player2_name} wins: {(1-prob)*100:.2f}%")
    winner = player1_name if prob >= 0.5 else player2_name
    print(f"✅ Predicted winner: {winner}\n")
    return prob

# -------------------------------
# 11. Example usage
# -------------------------------
# Replace these names with players in your dataset
predict_match("Giovanni Mpetshi Perricard", "Holger Rune", player_stats, scaler, w, b)
predict_match("Gabriel Diallo", "Zizou Bergs", player_stats, scaler, w, b)

import pickle
pickle.dump((w, b, scaler, player_stats), open("model_data.pkl", "wb"))
print("✅ Model saved to model_data.pkl")
