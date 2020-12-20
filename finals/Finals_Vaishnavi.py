import sys

import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

# import Assignment4_FeatureEngineering as a4


def main(file, response):
    df = pd.read_csv(file)
    print("Our response variable is" + "\n" + response)
    #
    df = df.rename(columns={response: "response"})
    print(df.columns)

    # We will also drop out the game_id, home_team_id, and away_team_id
    # Also dropping the SB_percent for both home and away games,
    # as the column is completely NULL
    # 'SB_percent_home','SB_percent_Away'

    df = df.drop(["home_team_id", "away_team_id", "game_id"], axis=1)
    df = df.dropna(axis=1, how="any")
    X = df.drop("response", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, df["response"], test_size=0.15, random_state=1234
    )
    pipeline = Pipeline([("Normalize", Normalizer()), ("lda", LDA(n_components=1))])
    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)

    # Generating the confusion matrix and accuracy score for the LDAClassifier
    cm = confusion_matrix(y_test, predict)
    print("Confusion Matrix- LDAClassifier" + "\n", cm)
    accuracy = accuracy_score(y_test, predict)
    print("Accuracy of LDAClassifier", accuracy)

    # Random Forest Classifier
    classifier = RandomForestClassifier(max_depth=5, random_state=1234)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Confusion Matrix- Random Forest Classifier")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Accuracy of Random Forest" + str(accuracy_score(y_test, y_pred)))

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    y_pred = logreg.predict(X_test)
    print(
        "Accuracy of logistic regression classifier on test set: {:.2f}".format(
            logreg.score(X_test, y_test)
        )
    )

    # Calculation the F-1, precision and recall
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig("Log_ROC")
    plt.show()

    # SVM
    clf = svm.SVC()
    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Calculation the F-1, precision and recall
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    file = "features.csv"
    response = "Home_team_wins"
    sys.exit(main(file, response))
