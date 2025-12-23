import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def main():
    # 1. Read Dataset
    df = pd.read_csv("epl_final.csv")

    # 2. Select Features
    X = df[
        [
            "FullTimeHomeGoals",
            "FullTimeAwayGoals",
            "HomeShots",
            "AwayShots",
            "HomeShotsOnTarget",
            "AwayShotsOnTarget",
        ]
    ]

    y = df["FullTimeResult"].map({"H": 3, "D": 1, "A": 0})

    # 4. Random Forest with K-Fold Cross Validation
    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf_cv.fit(X_train, y_train)
        y_pred = rf_cv.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores.append(acc)

    print("=== Random Forest with K-Fold CV ===")
    print("Accuracy per fold:", cv_scores)
    print("Mean Accuracy:", sum(cv_scores) / len(cv_scores))

    # 5. Random Forest without Cross Validation (Train/Test split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc_no_cv = accuracy_score(y_test, y_pred)
    cm_no_cv = confusion_matrix(y_test, y_pred)

    print("\n=== Random Forest without CV ===")
    print("Accuracy:", acc_no_cv)
    print("Confusion Matrix:\n", cm_no_cv)

    # 6. Feature Importance
    importances = rf.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Random Forest")
    plt.show()

    y = [acc_no_cv, sum(cv_scores) / len(cv_scores)]
    x = [0.5, 1]
    plt.bar(x, y)
    plt.show()


if __name__ == "__main__":
    main()
