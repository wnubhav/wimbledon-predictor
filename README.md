# Wimbledon Predictor 🎾

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnubhav/wimbledon-predictor/blob/main/wimbledon.ipynb)

A machine learning project that predicts the winner of the Wimbledon 2025 Men's Singles tournament using XGBoost trained on 2022–2024 match data, with a bracket simulation run over 1000 iterations.

---

## Overview

The project trains a binary classifier on historical Wimbledon Men's Singles data (2022, 2023, 2024) to predict match outcomes, then simulates the full 128-player bracket repeatedly to estimate each player's probability of winning the tournament.

---

## How It Works

**1. Data**
Match-level data from Wimbledon Men's Singles 2022–2024 is loaded from three CSVs and concatenated into a single dataframe. Only completed matches are used. Set scores (games won/lost per set) are retained; retired/walkover matches are filtered out.

**2. Feature Engineering**
The following features are constructed per match:
- `Player1_Rank`, `Player2_Rank` - ATP rankings at time of match
- `Player1_Pts`, `Player2_Pts` - ATP ranking points
- `Rank_Diff` - difference in rankings (Winner rank − Loser rank)
- `Points_Diff` - difference in ranking points
- `Winner_Win_Rate`, `Loser_Win_Rate` - historical win rate computed from the dataset

To remove label bias, Player 1 and Player 2 are assigned randomly per match, and the target is `Player1_Won` (1 or 0).

**3. Train/Validation Split**
- **Train**: 2022 and 2023 matches
- **Validation**: 2024 matches

**4. Model**
An XGBoost classifier (`binary:logistic`) is trained and then tuned with `GridSearchCV` (5-fold CV) over:
- `learning_rate`: [0.01, 0.1, 0.3]
- `max_depth`: [3, 5, 7]
- `n_estimators`: [100, 200, 300]

The best estimator from the grid search is used for all subsequent simulation.

**5. Bracket Simulation**
Players who appeared in Wimbledon 2024 are used as the 2025 draw proxy. The top 128 by ATP rank are seeded into the bracket. Each match is simulated by feeding player features into the model's `predict_proba`, and the winner is sampled from that probability. The tournament is simulated **1000 times**, and the predicted champion is the player who wins most frequently.

**6. Output**
- Single-run predicted champion (displayed in the notebook)
- Frequency table of winners across 1000 simulations
- Bar chart of the top 10 most frequently predicted champions

---

## Project Structure

```
wimbledon-predictor/
│
├── wimbledon.ipynb               # Main notebook: data loading, feature engineering,
│                                 # model training, hyperparameter tuning, simulation
└── README.md
```

**Data files required (not included in repo):**
```
Wimbledon_Mens_2022.csv
Wimbledon_Mens_2023.csv
Wimbledon_Mens_2024.csv
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, feature construction |
| `scikit-learn` | Train/val split, GridSearchCV, evaluation metrics |
| `xgboost` | Match outcome classification |
| `matplotlib` | Winner frequency bar chart |

---

## Setup

**Clone the repo**
```bash
git clone https://github.com/wnubhav/wimbledon-predictor.git
cd wimbledon-predictor
```

**Install dependencies**
```bash
pip install xgboost pandas numpy scikit-learn matplotlib jupyter
```

**Add the data files**

Place the three CSV files (`Wimbledon_Mens_2022.csv`, `Wimbledon_Mens_2023.csv`, `Wimbledon_Mens_2024.csv`) in the working directory (or `/content/` if running on Colab).

**Run the notebook**
```bash
jupyter notebook wimbledon.ipynb
```

Or open directly in Google Colab using the badge above.

---

