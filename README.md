# Predict.py
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# 1. SETUP - These will be stored securely in GitHub Secrets
API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def get_data():
    # Fetching recent match results (Example: English Premier League)
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {'X-Auth-Token': API_KEY}
    response = requests.get(url, headers=headers).json()
    return response['matches']

def train_and_predict():
    data = get_data()
    df = pd.json_normalize(data)
    
    # Simple Feature Engineering: Convert wins/losses to numbers
    # 0 = Draw, 1 = Home Win, 2 = Away Win
    df['target'] = df['score.winner'].map({'HOME_TEAM': 1, 'AWAY_TEAM': 2, 'DRAW': 0})
    
    # We filter for completed matches to train the AI
    train_df = df[df['status'] == 'FINISHED'].dropna(subset=['target'])
    
    # Features: Using Home/Away Team IDs as basic indicators
    X = train_df[['homeTeam.id', 'awayTeam.id']]
    y = train_df['target']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Predict upcoming matches
    upcoming = df[df['status'] == 'TIMED']
    if not upcoming.empty:
        predictions = model.predict(upcoming[['homeTeam.id', 'awayTeam.id']])
        probs = model.predict_proba(upcoming[['homeTeam.id', 'awayTeam.id']])
        
        message = "🤖 AI PREDICTIONS FOR TODAY:\n"
        for i, pred in enumerate(predictions):
            confidence = max(probs[i]) * 100
            if confidence > 70:  # Only alert if confidence is high
                home = upcoming.iloc[i]['homeTeam.name']
                away = upcoming.iloc[i]['awayTeam.name']
                res = "Home Win" if pred == 1 else "Away Win" if pred == 2 else "Draw"
                message += f"⚽ {home} vs {away}: {res} ({confidence:.1f}%)\n"
        
        send_telegram(message)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={text}"
    requests.get(url)

if __name__ == "__main__":
    train_and_predict()
    
