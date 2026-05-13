import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import joblib


def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.replace({"negtaive": "negative"})
    y = y.str.lower()
    return X, y


def split_data(X, y):
    idx = np.arange(len(X))

    return train_test_split(
        X, y, idx,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def run_linear_svc(X_train, X_test, y_train, y_test):
    print("\n--- Linear SVC (F1 Optimized) ---")

    model = LinearSVC()

    params = {
        "C": [0.01, 0.1, 1, 5, 10]
    }

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    f1 = f1_score(y_test, preds, average="weighted")

    print("Best Params:", grid.best_params_)
    print("F1 Score:", f1)
    print(classification_report(y_test, preds))

    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return best_model, preds, f1


def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n--- Logistic Regression (F1 Optimized) ---")

    model = LogisticRegression(max_iter=2000)

    params = {
        "C": [0.01, 0.1, 1, 5, 10],
        "solver": ["lbfgs", "saga"]
    }

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    f1 = f1_score(y_test, preds, average="weighted")

    print("Best Params:", grid.best_params_)
    print("F1 Score:", f1)
    print(classification_report(y_test, preds))

    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return best_model, preds, f1


def error_analysis(X_test, y_test, preds, idx_test):
    print("\n--- Error Analysis ---")

    results = pd.DataFrame({
        "index": idx_test,
        "true": y_test.values,
        "pred": preds
    })

    wrong = results[results["true"] != results["pred"]]

    print("Wrong predictions:", len(wrong))
    print(wrong["true"].value_counts())


def run_on_dataset(name, file_path):
    print(f"Running on {name}")

    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test, _, idx_test = split_data(X, y)

    svc_model, svc_preds, svc_f1 = run_linear_svc(X_train, X_test, y_train, y_test)

    lr_model, lr_preds, lr_f1 = run_logistic_regression(X_train, X_test, y_train, y_test)

    if svc_f1 > lr_f1:
        best_model = svc_model
        best_f1 = svc_f1
        best_name = "LinearSVC"
    else:
        best_model = lr_model
        best_f1 = lr_f1
        best_name = "LogisticRegression"

    error_analysis(X_test, y_test, svc_preds, idx_test)

    return name, best_model, best_f1


def main():

    datasets = {
        "scheme1_bow_model": "data/scheme1/scheme1_bow_labeled.csv",
        "scheme2_bow_model": "data/scheme2/scheme2_bow_labeled.csv",
        "scheme3_bow_model": "data/scheme3/scheme3_bow_labeled.csv",
#        "scheme1_glove": "scheme1_glove.csv",
#        "scheme2_glove": "scheme2_glove.csv",
#        "scheme3_glove": "scheme3_glove.csv",
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, path in datasets.items():

        ds_name, model, f1 = run_on_dataset(name, path)

        print(f"{ds_name} F1 = {f1}")

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = ds_name

    print("\n====================")
    print("BEST MODEL (F1)")
    print("====================")
    print("Dataset:", best_name)
    print("F1:", best_score)

    joblib.dump(best_model, "BEST_MODEL_F1_trial.pkl")
    print("Saved BEST_MODEL_F1_trial.pkl")


if __name__ == "__main__":
    main()