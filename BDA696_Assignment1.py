import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def main():
    columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Species"]
    iris_data = pd.read_csv("Iris.data", names=columns)
    print(iris_data.head(10))
    print(iris_data.info())

    # getting the number of missing values
    missing = iris_data.isnull().sum()
    print(missing)

    # calculating the summary statistics
    print("Mean statistics:", np.mean(iris_data))
    print("Maximum value:", np.max(iris_data))
    print("Minimum value:", np.min(iris_data))

    # Calculating quantiles
    iris_np = np.array(iris_data)
    print("25 percent- ", np.quantile(iris_np[:, :-1], 0.25, axis=0))
    print("50 percent-", np.quantile(iris_np[:, :-1], 0.50, axis=0))
    print("75 percent-", np.quantile(iris_np[:, :-1], 0.75, axis=0))

    # Visualizing the data
    plot1 = px.violin(
        iris_data,
        y="Sepal_Length",
        x="Species",
        color="Species",
        hover_data=iris_data.columns,
        title="Violin plot to visualize Sepal Length ",
    )
    plot1.show()

    plot2 = px.scatter_3d(
        iris_data,
        x="Sepal_Width",
        y="Sepal_Length",
        z="Petal_Width",
        color="Species",
        hover_data=iris_data.columns,
        title="3d Scatter plot between Sepal Width, " "Sepal length and Petal width",
    )
    plot2.show()

    plot3 = px.histogram(
        iris_data,
        x="Sepal_Length",
        y="Sepal_Width",
        color="Species",
        title="Distribution of Sepal length and Sepal width with respect to species",
    )
    plot3.show()

    iris_data.plot(kind="scatter", x="Sepal_Length", y="Petal_Length")
    plt.title("Scatter plot between Sepal Length and petal Length")
    plt.show()

    sns.set_style("whitegrid")
    sns.FacetGrid(iris_data, hue="Species", height=6).map(
        plt.scatter, "Sepal_Length", "Petal_Length"
    ).add_legend()
    plt.title(
        "Scatter plot between Sepal Length and Petal Length, with different species"
    )
    plt.show()

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.iloc[:, :-1].values,
        iris_data["Species"],
        test_size=0.2,
        random_state=1234,
    )
    # Making a Pipeline for Normalizing the data and fitting RandomForestClassifier
    pipeline = Pipeline(
        [("Normalize", Normalizer()), ("rf", RandomForestClassifier(random_state=1234))]
    )
    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)

    # Generating the confusion matrix and accuracy score for the RandomForestClassifier
    cm = confusion_matrix(y_test, predict)
    print("Confusion Matrix- RandomForestClassifier", cm)
    accuracy = accuracy_score(y_test, predict)
    print("Accuracy of RandomForestClassifier", accuracy)

    # Making a pipeline for Normalizing the data and fitting LDAClassifier
    pipeline = Pipeline([("Normalize", Normalizer()), ("lda", LDA(n_components=1))])
    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)
    # Generating the confusion matrix and accuracy score for the LDAClassifier
    cm = confusion_matrix(y_test, predict)
    print("Confusion Matrix- LDAClassifier", cm)
    accuracy = accuracy_score(y_test, predict)
    print("Accuracy of LDAClassifier", accuracy)


if __name__ == "__main__":
    sys.exit(main())
