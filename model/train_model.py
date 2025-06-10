import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Simulação de dataset
os.makedirs("data", exist_ok=True)
data = {
    "home_team_rank": [1, 5, 8, 3, 10, 2, 7, 6],
    "away_team_rank": [10, 6, 2, 7, 1, 4, 9, 8],
    "home_win_rate": [0.85, 0.60, 0.50, 0.75, 0.30, 0.80, 0.55, 0.58],
    "away_win_rate": [0.40, 0.45, 0.70, 0.35, 0.90, 0.60, 0.42, 0.55],
    "goals_home_avg": [2.1, 1.8, 1.3, 2.0, 0.9, 2.2, 1.4, 1.5],
    "goals_away_avg": [1.0, 1.2, 2.0, 1.1, 2.5, 1.4, 1.0, 1.3],
    "result": ["home_win", "home_win", "away_win", "draw", "away_win", "home_win", "draw", "draw"]
}
df = pd.DataFrame(data)
df.to_csv("data/matches.csv", index=False)

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/match_model.pkl")
print("Modelo salvo com sucesso.")