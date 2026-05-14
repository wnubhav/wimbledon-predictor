# Wimbledon Predictor 🎾

A machine learning project that predicts the winner of the Wimbledon 2025 Men's Singles tournament using XGBoost and full bracket simulation.

---

## Overview

This project builds a match outcome predictor trained on historical ATP match data, then runs a bracket simulation across the Wimbledon 2025 draw to forecast the likely champion. Each match in the bracket is predicted independently using player statistics, surface performance, and ranking features , and the simulation propagates winners round by round all the way to the final.

---

## How It Works

**1. Feature Engineering**
Player and match features are constructed from historical ATP data, including:
- ATP ranking and ranking points
- Grass-court specific win rates
- Head-to-head records
- Recent form (win % over last N matches)
- Serve and return stats (aces, double faults, break points)

**2. Model**
An XGBoost classifier is trained on historical Wimbledon and grass-court match data to predict the probability of a player winning a given match. The model outputs win probabilities for each player in a head-to-head.

**3. Bracket Simulation**
The full 128-player Wimbledon 2025 draw is seeded into the bracket. The trained model then simulates each round, selecting winners based on predicted probabilities. This is run across multiple iterations (Monte Carlo style) to produce win probability distributions for each player across rounds.

**4. Output**
Final predicted winner and win probabilities per player, along with round-by-round survival rates.

---

## Project Structure

```
wimbledon-predictor/
│
├── wimbledon.ipynb      # Main notebook: EDA, feature engineering, model training, bracket simulation
└── README.md
```

---

## Tech Stack

- **Python** (Jupyter Notebook)
- **XGBoost** - match outcome classification
- **pandas / NumPy** - data processing and feature construction
- **scikit-learn** - preprocessing, train/test split, evaluation metrics
- **Matplotlib / Seaborn** - visualizations

---

## Setup

**Clone the repo**
```bash
git clone https://github.com/wnubhav/wimbledon-predictor.git
cd wimbledon-predictor
```

**Install dependencies**
```bash
pip install xgboost pandas numpy scikit-learn matplotlib seaborn jupyter
```

**Run the notebook**
```bash
jupyter notebook wimbledon.ipynb
```

---

## Results

The model simulates the full Wimbledon 2025 bracket and outputs predicted win probabilities for each player. Results and visualizations are in the notebook.

---
