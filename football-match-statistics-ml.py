import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("epl_final.csv")

    features = [
        "HalfTimeHomeGoals",
        "HalfTimeAwayGoals",
        "HomeShots",
        "AwayShots",
        "HomeShotsOnTarget",
        "AwayShotsOnTarget",
        "HomeCorners",
        "AwayCorners",
        "HomeFouls",
        "AwayFouls",
        "HomeRedCards",
        "AwayRedCards",
    ]

    X = df[features]
    y = df["FullTimeResult"]  # target

    # ----- Random Forest Classifier without Cross Validation -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    classifier_rf = RandomForestClassifier(
        n_estimators=300, max_depth=7, random_state=42
    )
    classifier_rf.fit(X_train, y_train)
    y_pred = classifier_rf.predict(X_test)

    acc_no_cv = metrics.accuracy_score(y_test, y_pred)

    classification_rep = metrics.classification_report(y_test, y_pred)

    print(f"\nAccuracy (without CV) :\n {acc_no_cv}")
    print("\nClassification Report :\n", classification_rep)

    # ----- Random Forest Classifier with K-Fold Cross Validation ------
    print(" --- Random Forest with K-Fold Cross Validation ---")
    rf_cv = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf_cv.fit(X_train, y_train)
        y_pred = rf_cv.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        cv_scores.append(acc)
        print(
            "\nClassification Report:\n", metrics.classification_report(y_test, y_pred)
        )

    mean_accuracy = sum(cv_scores) / len(cv_scores)

    print("\nAccuracy per fold :\n", cv_scores)
    print("\nMean Accuracy : \n", mean_accuracy)

    importances = classifier_rf.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Random Forest")
    plt.show()


if __name__ == "__main__":
    main()
