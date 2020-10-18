import os
import sys

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import statsmodels
import statsmodels.api
from plotly import express as px
from plotly import figure_factory, graph_objects
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix


def main(file, response):
    df = pd.read_csv(file)
    print("Our response variable is" + "\n" + response)
    df = df.dropna(axis=1, how="any")
    df = df.rename(columns={response: "response"})
    y = df.response.values
    pred = df.drop("response", axis=1)

    # Let's make a folder to store our plots
    if not os.path.exists("assignment4/graph"):
        os.makedirs("assignment4/graph")

    # Let's check if the response variable is Continuous or Boolean
    response_var_type = continuous_or_boolean(df)

    # If the variable is categorical we will use Regressor
    if response_var_type == "Categorical":
        regressor = RandomForestRegressor(
            n_estimators=50, oob_score=True, random_state=12034
        )
        regressor.fit(pred, y)

        # Computing the variable importance
        importance = regressor.feature_importances_

    # If the variable is boolean we will use Classifier
    else:
        df["response"] = df["response"].astype("category")
        df["response"] = df["response"].cat.codes
        classifier = RandomForestClassifier(
            n_estimators=50, oob_score=True, random_state=1234
        )
        classifier.fit(pred, y)

        importance = classifier.feature_importances_

    # Defining the dataframe we want to display on our HTML page
    result = pd.DataFrame()
    result["Variable"] = pred.columns

    t_value = []
    p_value = []
    m_plot = []
    cat_con = []
    file = []
    mean_unweighted = []
    mean_weighted = []
    # msd_plot=[]
    for var in pred.columns:
        # Let's first calculate p-value and t-value
        if response_var_type == "Categorical":
            y = df["response"]
            predictor = statsmodels.api.add_constant(df[var])
            linear_regression_model = statsmodels.api.OLS(y, predictor)
            linear_regression_model_fitted = linear_regression_model.fit()
            print(linear_regression_model_fitted.summary())

            # Getting the t_value and p_value using the model
            tvalue = round(linear_regression_model_fitted.tvalues[1], 6)
            pvalue = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

            # Appending it to the list
            t_value.append(tvalue)
            p_value.append(pvalue)

            # Plotting the figure
            figure = px.scatter(x=df[var], y=y, trendline="ols")
            figure.update_layout(
                title=f"Variable: {var}: (t-value={tvalue}) (p-value={pvalue})",
                xaxis_title=f"Variable: {var}",
                yaxis_title="y",
            )

            figure.write_html(
                file=f"assignment4/graph/ranking_{var}.html", include_plotlyjs="cdn"
            )
            filepath = "assignment4/graph/ranking_" + var + ".html"
            m_plot.append("<a href=" + filepath + ">" + filepath + "</a>")

        else:
            y = df["response"]
            predictor = statsmodels.api.add_constant(df[var])
            logistic_regression_model = statsmodels.api.Logit(y, predictor)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            print(logistic_regression_model_fitted.summary())

            # Getting the t_value and p_value using the model
            tvalue = round(logistic_regression_model_fitted.tvalues[1], 6)
            pvalue = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

            # Appending it to the list
            t_value.append(tvalue)
            p_value.append(pvalue)

            # Plotting the figure
            file_name = "assignment4/graph/ranking_" + var + ".html"
            m_plot.append("<a href=" + file_name + ">" + file_name + "</a>")
            # Plot the figure
            figure = px.scatter(x=df[var], y=y, trendline="ols")
            figure.update_layout(
                title=f"Variable: {var}: (t-value={tvalue}) (p-value={pvalue})",
                xaxis_title=f"Variable: {var}",
                yaxis_title="y",
            )
            figure.write_html(
                file=f"assignment4/graph/ranking_{var}.html", include_plotlyjs="cdn"
            )

        if cat_or_bool(pred[var]) and response_var_type == "Boolean":
            fpath = "assignment4/graph/cat_cat_heatmap" + var + ".html"

            cm = confusion_matrix(var, df["response"])
            plot = graph_objects.Figure(data=graph_objects.Heatmap(z=cm, zmax=cm.max()))

            # Let's give out the titles and other details to our graph
            plot.update_layout(
                title="Categorical Predictor by Categorical Response ",
                xaxis_title="Response",
                yaxis_title="Predictor",
            )

            plot.write_html(
                file=fpath,
                include_plotlyjs="cdn",
            )

            cat_con.append("Categorical")
            file.append("")

        elif not cat_or_bool(pred[var]) and response_var_type == "Boolean":
            fpath = "assignment4/graph/cat_con_dist" + var + ".html"
            cat_con_dist(df, var, fpath)
            cat_con.append("Continuous")
            file.append("<a href=" + fpath + ">" + fpath + "</a>")

        elif cat_or_bool(pred[var]) and response_var_type != "Boolean":
            fpath = "assignment4/graph/con_cat_violin" + var + ".html"
            cont_cat_violin(df, var, fpath)
            cat_con.append("Categorical")
            file.append("<a href=" + fpath + ">" + fpath + "</a>")

        elif not cat_or_bool(pred[var]) and response_var_type != "Boolean":
            fpath = "assignment4/graph/con_con_scatter" + var + ".html"

            plot1 = df.plot(kind="scatter", x=var, y=df["response"])
            plt.xlabel("Predictor")
            plt.ylabel("Response")
            plt.title("Continuous/Continuous")
            plot1.write_html(
                file=fpath,
                include_plotlyjs="cdn",
            )

            cat_con.append("Continuous")
            file.append("<a href=" + fpath + ">" + fpath + "</a>")

        # Let's calculate weighted and unweighted mean squared difference.
        pp = df.response.sum() / len(df)
        if cat_or_bool(pred[var]):
            output = pd.DataFrame()
            bin1 = pd.DataFrame({"pred": df[var], "res": df["response"]}).groupby(
                df[var]
            )
            output["total"] = bin1.count().res
            population_pr = output["total"] / len(df)
            output["mean1"] = bin1.sum().res / output.total
            output["dif"] = (output.mean1 - pp) ** 2
            output["dif_weighted"] = output.dif * population_pr

            unweighted = output["dif"].sum()
            weighted = output["dif_weighted"].sum()

        else:
            output = pd.DataFrame()
            bin1 = pd.DataFrame(
                {"pred": df[var], "res": df["response"], "Bucket": pd.qcut(df[var], 10)}
            )
            group = bin1.groupby("Bucket", as_index=True)
            output["mean2"] = bin1.mean().pred
            output["minimum"] = group.min().pred
            output["max"] = group.max().pred
            output["population_mean"] = pp
            output["total"] = group.count().res
            population_pr = output["total"] / len(df)
            output["mean1"] = group.sum().res / output.total
            output["dif"] = (output.mean1 - pp) ** 2
            output["dif_weighted"] = output.dif * population_pr

            unweighted = output["dif"].sum()
            weighted = output["dif_weighted"].sum()

            diff_mean_plot(output)
            # msd_plot.append(plotmsd)
        # Appending the values
        mean_unweighted.append(unweighted)
        mean_weighted.append(weighted)

    # Assigning everything to a dataframe
    result["cat_con"] = cat_con
    result["t-value"] = t_value
    result["p-value"] = p_value
    result["m-plot"] = m_plot
    result["graphs"] = file
    result["Variable_importance"] = importance
    result["mean_unweighted"] = mean_unweighted
    result["mean_weighted"] = mean_weighted
    # result['msd_plot'] = msd_plot

    print(result)
    result.to_html("Bihare_Vaishnavi_Assignment4.html", render_links=True, escape=False)


def continuous_or_boolean(df):
    if df.response.nunique() > 2:
        return "Continuous"
    else:
        return "Boolean"


def cat_or_bool(predictor):
    if predictor.dtypes == "object":
        return True
    elif predictor.nunique() / predictor.count() < 0.05:
        return True
    else:
        return False


def cont_cat_violin(df, col, file_name):
    # Grouping
    groups = ["0", "1"]
    plot = graph_objects.Figure()
    label0 = df[df["target"] == 0][col]
    label1 = df[df["target"] == 1][col]
    for curr_hist, curr_group in zip([label0, label1], groups):
        plot.add_trace(
            graph_objects.Violin(
                x=numpy.repeat(curr_group, len(df)),
                y=curr_hist,
                name=int(curr_group),
            )
        )
    # Let's give out the titles and other details to our graph
    plot.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    plot.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )


def cat_con_dist(df, col, file_name):
    # Grouping
    groups = ["0", "1"]
    label0 = df[df["response"] == 0][col]
    label1 = df[df["response"] == 1][col]

    plot = figure_factory.create_distplot([label0, label1], groups)
    plot.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    plot.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )


# https://plotly.com/python/multiple-axes/ ##very helpful
def diff_mean_plot(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        graph_objects.Bar(x=df["mean1"], y=df["total"]),
        secondary_y=False,
    )

    fig.add_trace(
        graph_objects.Scatter(
            x=df["mean2"], y=df["mean1"], line=dict(color="green"), name="mean"
        ),
        secondary_y=True,
    )
    fig.add_trace(
        graph_objects.Scatter(
            x=df["mean2"],
            y=df["population_mean"],
            line=dict(color="red"),
            name="PopulationMean",
        ),
        secondary_y=True,
    )
    fig.update_layout(height=500, width=850, title_text="Difference in mean")
    fig.write_html(
        file="assignment4/graph/mean_diff.html",
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    # file = "data.csv"
    # response = "diagnosis"
    file = sys.argv[1]
    response = sys.argv[2]
    sys.exit(main(file, response))
