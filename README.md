def calculate_form(df):
    # 1. Assign points based on the winner
    def get_pts(row, team_id):
        if row['score.winner'] == 'DRAW': return 1
        if (row['score.winner'] == 'HOME_TEAM' and row['homeTeam.id'] == team_id) or \
           (row['score.winner'] == 'AWAY_TEAM' and row['awayTeam.id'] == team_id):
            return 3
        return 0

    # 2. Create a "Form" column for each team
    # This part groups by team and calculates the average points of the last 5 games
    df = df.sort_values('utcDate')
    
    # We calculate home_form and away_form separately
    df['home_pts'] = df.apply(lambda x: get_pts(x, x['homeTeam.id']), axis=1)
    df['away_pts'] = df.apply(lambda x: get_pts(x, x['awayTeam.id']), axis=1)
    
    # Rolling average of the last 5 games
    df['home_form'] = df.groupby('homeTeam.id')['home_pts'].transform(lambda x: x.shift().rolling(5).mean())
    df['away_form'] = df.groupby('awayTeam.id')['away_pts'].transform(lambda x: x.shift().rolling(5).mean())
    
    return df.fillna(0) # Fill early season games (where no 5-game history exists) with 0

def train_and_predict():
    raw_data = get_data()
    df = pd.json_normalize(raw_data)
    
    # Apply our new Form logic
    df = calculate_form(df)
    
    df['target'] = df['score.winner'].map({'HOME_TEAM': 1, 'AWAY_TEAM': 2, 'DRAW': 0})
    train_df = df[df['status'] == 'FINISHED'].dropna(subset=['target'])
    
    # NEW: We now include 'home_form' and 'away_form' as inputs!
    features = ['homeTeam.id', 'awayTeam.id', 'home_form', 'away_form']
    X = train_df[features]
    y = train_df['target']
    
    model = RandomForestClassifier(n_estimators=200) # Increased trees for better precision
    model.fit(X, y)
    
    # ... rest of the prediction and Telegram logic remains the same
    
