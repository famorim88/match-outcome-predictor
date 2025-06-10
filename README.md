# ⚽ Match Outcome Predictor

Preveja o resultado de partidas de futebol (vitória do mandante, empate, vitória do visitante) com base em estatísticas do histórico recente dos times.

## 🛠 Tecnologias
- Python 3.10
- FastAPI
- Scikit-learn
- RandomForestClassifier
- Docker

## 🔢 Features
- Rank dos times
- Taxa de vitórias recentes
- Gols médios marcados

## 🧪 Dataset
Simulado (mas pode ser integrado com dados da API do futebol, como SportsDataIO ou Footystats).

## 🚀 Como usar

### 1. Treinar modelo
```bash
python train/train_model.py
