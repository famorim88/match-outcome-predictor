from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
model = joblib.load("model/match_model.pkl")

API_KEY = os.getenv("API_FUTEBOL_KEY")
API_BASE_URL = "https://api.api-futebol.com.br/v1"

class Match(BaseModel):
    home_team_rank: int
    away_team_rank: int
    home_win_rate: float
    away_win_rate: float
    goals_home_avg: float
    goals_away_avg: float

@app.post("/predict")
def predict_match(match: Match):
    data = pd.DataFrame([match.dict()])
    pred = model.predict(data)[0]
    return {"predicted_outcome": pred}

@app.get("/data/round/{round_number}")
def get_serie_a_round(round_number: int):
    url = f"{API_BASE_URL}/campeonatos/10/rodadas/{round_number}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to fetch data", "status_code": response.status_code}