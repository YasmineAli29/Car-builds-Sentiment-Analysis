import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def load_data(file_path):
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    y = y.replace({"negtaive": "negative"})
    y = y.str.lower()

    return X, y



def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def run_multinomial_nb(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds), classification_report(y_test, preds)


def run_gaussian_nb(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds), classification_report(y_test, preds)


def run_logreg(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds), classification_report(y_test, preds)


def run_svc(X_train, X_test, y_train, y_test):
    model = LinearSVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds), classification_report(y_test, preds)


def run_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds), classification_report(y_test, preds)


def run_on_dataset(name, file_path):

    print(f"Running on {name}\n")
    
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    is_glove = "glove" in name.lower()


    if not is_glove:
        print("\n--- Multinomial Naive Bayes (BoW only) ---")
        acc, report = run_multinomial_nb(X_train, X_test, y_train, y_test)
        print("Accuracy:", acc)
        print(report)

    print("\n--- Gaussian Naive Bayes ---")
    acc, report = run_gaussian_nb(X_train, X_test, y_train, y_test)
    print("Accuracy:", acc)
    print(report)

    print("\n--- Logistic Regression ---")
    acc, report = run_logreg(X_train, X_test, y_train, y_test)
    print("Accuracy:", acc)
    print(report)

    print("\n--- Linear SVC ---")
    acc, report = run_svc(X_train, X_test, y_train, y_test)
    print("Accuracy:", acc)
    print(report)

    print("\n--- Decision Tree ---")
    acc, report = run_decision_tree(X_train, X_test, y_train, y_test)
    print("Accuracy:", acc)
    print(report)



def main():
    datasets = {
        "scheme1_bow_model": "data/scheme1/scheme1_bow_labeled.csv",
        "scheme2_bow_model": "data/scheme2/scheme2_bow_labeled.csv",
        "scheme3_bow_model": "data/scheme3/scheme3_bow_labeled.csv",
        "scheme1_glove_model": "data/scheme1/scheme1_glove_labeled.csv",
        "scheme2_glove_model": "data/scheme2/scheme2_glove_labeled.csv",
        "scheme3_glove_model": "data/scheme3/scheme3_glove_labeled.csv",
    }

    for name, path in datasets.items():
        run_on_dataset(name, path)


if __name__ == "__main__":
    main()