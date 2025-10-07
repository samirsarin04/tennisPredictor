# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle

st.set_page_config(page_title="🎾 Tennis Match Predictor", layout="centered")

# -------------------------------
# 1. Load model + data
# -------------------------------
@st.cache_resource
def load_model():
    with open("model_data.pkl", "rb") as f:
        w, b, scaler, player_stats = pickle.load(f)
    return w, b, scaler, player_stats

w, b, scaler, player_stats = load_model()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------------------
# 2. Prediction function
# -------------------------------
def predict_match(player1_name, player2_name):
    if player1_name not in player_stats.index or player2_name not in player_stats.index:
        return None, f"❌ One or both players not found in dataset."
    p1 = player_stats.loc[player1_name]
    p2 = player_stats.loc[player2_name]
    X = np.array([[
        p2['rank'] - p1['rank'],
        p1['age'] - p2['age'],
        p1['ht'] - p2['ht'],
        p1['ace'] - p2['ace'],
        p2['df'] - p1['df'],
        p1['bp_ratio'] - p2['bp_ratio'],
        p1['rank_points'] - p2['rank_points']
    ]])
    X_scaled = scaler.transform(X)
    prob = sigmoid(np.dot(X_scaled, w) + b)[0]
    return prob, None

# -------------------------------
# 3. Optional: Live Tennis Data
# -------------------------------
def get_live_matches():
    try:
        url = "https://api-tennis.p.rapidapi.com/matches/live"
        headers = {
            "X-RapidAPI-Key": st.secrets["RAPIDAPI_KEY"],
            "X-RapidAPI-Host": "api-tennis.p.rapidapi.com"
        }
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json().get("response", [])
        matches = []
        for m in data:
            p1 = m.get("player1", {}).get("name")
            p2 = m.get("player2", {}).get("name")
            score = m.get("score", "N/A")
            if p1 and p2:
                matches.append({"p1": p1, "p2": p2, "score": score})
        return matches
    except Exception as e:
        return []

# -------------------------------
# 4. UI
# -------------------------------
st.title("🎾 Tennis Match Predictor")
st.caption("Predict the winner between two ATP players using data-based modeling.")

tab1, tab2 = st.tabs(["🔮 Predict Match", "📡 Live Matches"])

with tab1:
    st.subheader("Manual Prediction")
    player_names = sorted(player_stats.index.tolist())

    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", player_names, index=0)
    with col2:
        player2 = st.selectbox("Player 2", player_names, index=1)

    if st.button("Predict"):
        prob, err = predict_match(player1, player2)
        if err:
            st.error(err)
        else:
            st.success(f"Probability {player1} wins: {prob*100:.2f}%")
            st.info(f"Probability {player2} wins: {(1-prob)*100:.2f}%")
            winner = player1 if prob >= 0.5 else player2
            st.markdown(f"🏆 **Predicted Winner:** {winner}")

with tab2:
    st.subheader("Live Match Predictions")
    st.caption("Fetches current live matches (via RapidAPI).")

    if "RAPIDAPI_KEY" not in st.secrets:
        st.warning("Add your RapidAPI key to Streamlit Cloud secrets to enable live data.")
    else:
        matches = get_live_matches()
        if matches:
            for m in matches:
                p1, p2, score = m["p1"], m["p2"], m["score"]
                prob, _ = predict_match(p1, p2)
                st.write(f"**{p1} vs {p2}**")
                st.write(f"🕹️ Score: {score}")
                if prob:
                    st.write(f"Predicted {p1} win chance: {prob*100:.2f}%")
                st.markdown("---")
        else:
            st.info("No live matches found or API not available right now.")
