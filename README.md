# English Premier League Match Outcome Prediction
Random Forest Classification with and without K-Fold Cross Validation

Project Description
- This project predicts the full-time match outcome of English Premier League (EPL) games using a Random Forest Classification model.
The prediction is performed from the home team’s perspective based on statistical match data.

---

Objective
- The goal of this project is to predict whether the home team wins, draws, or loses using match statistics and to compare model performance with and without cross-validation.

---

Dataset
- Dataset: English Premier League (EPL) Match Statistics
- Seasons: 2000–2025
- Source: Kaggle  
- File: `epl_final.csv`

---

Target Variable
- `FullTimeResult`
  - "H": Home Win
  - "D": Draw
  - "A": Away Win

Selected Features
- HalfTimeHomeGoals
- HalfTimeAwayGoals
- HomeShots
- AwayShots
- HomeShotsOnTarget
- AwayShotsOnTarget
- HomeCorners
- AwayCorners
- HomeFouls
- AwayFouls
- HomeRedCards
- AwayRedCards

---

Machine Learning Model
- Algorithm: Random Forest Classifier
- Number of trees (`n_estimators`): 300
- Maximum depth (`max_depth`): 7
- Random state: 42

Two evaluation strategies are used:
1. Train/Test Split (80/20) without cross-validation
2. 5-Fold Cross Validation to evaluate model generalization

---

Evaluation Metrics
The model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Feature Importance

Classification reports are generated for both:
- The test set (without CV)
- Each fold during cross-validation

---

Feature Importance
- Feature importance is extracted from the trained Random Forest model to analyze which match statistics contribute most to the prediction of match outcomes.

---

Tools & Libraries
- Pandas
- Scikit-learn
- Matplotlib
