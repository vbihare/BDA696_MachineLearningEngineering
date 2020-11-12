import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
from plotly import graph_objects
from scipy import stats

import Assignment4_FeatureEngineering as a4


def main(file, response):
    df = pd.read_csv(file)
    print("Our response variable is" + "\n" + response)
    df = df.dropna(axis=1, how="any")
    df = df.rename(columns={response: "response"})
    # y = df.response.values
    pred = df.drop("response", axis=1)

    # Let's make a folder to store our plots
    if not os.path.exists("midterm"):
        os.makedirs("midterm")

    # Let's check if the response variable is Continuous or Boolean
    response_var_type = continuous_or_boolean(df)

    if response_var_type == "Boolean":
        df["response"] = df["response"].astype("category")
        df["response"] = df["response"].cat.codes

    # Getting the categorical and continuous columns
    condata, catdata = cat_con_list(pred, df)
    print("The categorical columns are:" + "\n")
    print(catdata)
    print("The continuous columns are:" + "\n")
    print(condata)

    # Let's now make some plots
    column_plot = {}

    for cols in condata:
        cols_proper = cols.replace(" ", "-").replace("/", "-")
        if response_var_type == "Boolean":
            file_path = "con_predictor_cat_response_distplot" + cols_proper + ".html"
            a4.cat_con_dist(df, cols, file_path)
            path = "<a href=" + file_path + ">" + cols

            column_plot[cols] = path
        else:
            file_path = "con_predictor_con_response_scatterplot" + cols_proper + ".html"
            a4.cat_con_scatter(df, cols, file_path)
            path = "<a href=" + file_path + ">" + cols

            column_plot[cols] = path

    for cols in catdata:
        cols_proper = cols.replace(" ", "-").replace("/", "-")
        if response_var_type == "Boolean":
            file_path = "cat_predictor_cat_response_heatmap" + cols_proper + ".html"
            a4.cat_con_heatmap(df, cols, file_path)
            path = "<a href=" + file_path + ">" + cols

            column_plot[cols] = path
        else:
            file_path = "cat_predictor_con_response_violinplot" + cols_proper + ".html"
            a4.cat_con_violin(df, cols, file_path)
            path = "<a href=" + file_path + ">" + cols

            column_plot[cols] = path

    if not os.path.exists("midterm/correlation"):
        os.makedirs("midterm/correlation")

    con_con_cor, con_con_matrix, con_con_msd = con_con_corr(condata, df, column_plot)
    # Sorting and printing the correlation matrix
    print("Continuous-Continuous Correlation metrics")
    con_con_cor.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(con_con_cor)
    # Plotting the correlation matrix
    con_con_matrix = con_con_matrix.astype(float)
    plot = sns.heatmap(con_con_matrix, annot=False)
    corr1 = plot.get_figure()
    corr1.savefig("midterm/correlation/con_con_corr.png")
    plt.clf()

    with open("midterm/correlation_con_con.html", "w") as _file:
        _file.write(con_con_cor.to_html(render_links=True, escape=False))

    cat_cat_cor, cat_cat_matrix, cat_cat_msd = cat_cat_corr(catdata, df, column_plot)
    # Sorting and printing the correlation matrix
    print("Categorical-Categorical Correlation metrics")
    cat_cat_cor.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_cat_cor)
    cat_cat_matrix = cat_cat_matrix.astype(float)
    plot2 = sns.heatmap(cat_cat_matrix, annot=False)
    corr3 = plot2.get_figure()
    corr3.savefig("midterm/correlation/cat_cat_corr.png")
    plt.clf()

    with open("midterm/correlation_cat_cat.html", "w") as _file:
        _file.write(cat_cat_cor.to_html(render_links=True, escape=False))

    cat_con_cor, cat_con_matrix, cat_con_msd = cat_con_corr(
        catdata, condata, df, column_plot
    )
    # Sorting and printing the correlation matrix
    print("Categorical-Continuous Correlation metrics")
    cat_con_cor.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_con_cor)
    # Plotting the correlation matrix
    cat_con_matrix = cat_con_matrix.astype(float)
    plot1 = sns.heatmap(cat_con_matrix, annot=False)
    corr2 = plot1.get_figure()
    corr2.savefig("midterm/correlation/cat_con_corr.png")
    plt.clf()
    with open("midterm/correlation_cat_con.html", "w") as _file:
        _file.write(cat_con_cor.to_html(render_links=True, escape=False))

    with open("midterm/correlation/correlation_plot.html", "w") as _file:
        _file.write(
            "<h1> Continuous Continuous Plot </h1> "
            + "<img src='con_con_corr.png'"
            + "<h1> Categorical Continuous Plot </h1> "
            + "<img src = 'cat_con_corr.png'"
            + "<h1> Categorical Categorical Plot </h1>"
            + "<img src ='cat_cat_corr.png'"
        )
        # Brute-Force
        # Cont-Cont Diff of mean
        msd_contcont = con_con_msd.sort_values(by="MeanSqDiffW", ascending=False)
        print(msd_contcont)

        # Cat-Cont Diff of mean
        msd_catcont = con_con_msd.sort_values(by="MeanSqDiffW", ascending=False)
        print(msd_catcont)

        # Cat-Cat
        msd_catcat = cat_cat_msd.sort_values(by="MeanSqDiffW", ascending=False)
        print(msd_catcat)

        with open("midterm/BruteForce.html", "w") as _file:
            _file.write(
                msd_contcont.to_html(render_links=True, escape=False)
                + "<br>"
                + msd_catcont.to_html(render_links=True, escape=False)
                + "<br>"
                + msd_catcat.to_html(render_links=True, escape=False)
            )


def continuous_or_boolean(df):
    if df.response.nunique() > 2:
        return "Continuous"
    else:
        return "Boolean"


def cat_or_con(predictor):
    if predictor.dtypes == "object":
        return True
    elif predictor.nunique() < 5:
        return True
    else:
        return False


def cat_con_list(pred, df):
    catdata = []
    condata = []
    for cols in pred.columns:
        cols_proper = cols.replace(" ", "-").replace("/", "-")
        if cat_or_con(pred[cols]):
            df[cols] = df[cols].astype("category")
            df[cols] = df[cols].cat.codes
            catdata.append(cols_proper)
        else:
            condata.append(cols_proper)

    return condata, catdata


def con_con_corr(cont_data, df, correlation_plot):
    final_col = [
        "Cont_var1",
        "Cont_var2",
        "MeanSquareDiff",
        "MeanSquareDiffWeighted",
        "PLot",
    ]
    con_con_matrix = pd.DataFrame(index=cont_data, columns=cont_data)
    cols = ["Cont_var1", "Cont_var2", "Correlation"]
    cont_cont_corr = pd.DataFrame(columns=cols)
    msd_con_con = pd.DataFrame(columns=final_col)
    pop_prop_1 = df.response.sum() / len(df)
    for var in range(len(cont_data)):
        for var2 in range(var, len(cont_data)):
            if cont_data[var] != cont_data[var2]:
                pearsonr_value, _ = stats.pearsonr(
                    df[cont_data[var]], df[cont_data[var2]]
                )

                con_con_matrix.loc[cont_data[var]][cont_data[var2]] = pearsonr_value
                con_con_matrix.loc[cont_data[var]][cont_data[var2]] = pearsonr_value
                cont_cont_corr.append(
                    dict(
                        zip(
                            cols,
                            [
                                correlation_plot[cont_data[var]],
                                correlation_plot[cont_data[var2]],
                                pearsonr_value,
                            ],
                        )
                    ),
                    ignore_index=True,
                )
                weighted, unweighted, data = get_msd(
                    cont_data[var], cont_data[var2], df, pop_prop_1, 1
                )
                msd_con_con = msd_con_con.append(
                    dict(
                        zip(
                            final_col,
                            [
                                cont_data[var],
                                cont_data[var2],
                                weighted,
                                unweighted,
                                data,
                            ],
                        )
                    ),
                    ignore_index=True,
                )
    return cont_cont_corr, con_con_matrix, msd_con_con


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(categories, values):
    f_cat, _ = pd.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values.iloc[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


def cat_con_corr(cat_data, cont_data, df, correlation_plot):
    cat_con_matrix = pd.DataFrame(index=cat_data, columns=cont_data)
    pop_prop_1 = df.response.sum() / len(df)
    final_col = [
        "Cat_var1",
        "Cont_var2",
        "MeanSquareDiff",
        "MeanSquareDiffWeighted",
        "PLot",
    ]
    msd_cat_con = pd.DataFrame(columns=final_col)
    cols = ["Cat_var", "Con_var", "Correlation"]
    cat_con_cor = pd.DataFrame(columns=cols)
    for var in range(len(cat_data)):
        for var2 in range(len(cont_data)):
            cor_value = cat_cont_correlation_ratio(
                df[cat_data[var]], df[cont_data[var2]]
            )
            cat_con_matrix.loc[cat_data[var]][cont_data[var2]] = cor_value
            cat_con_cor = cat_con_cor.append(
                dict(
                    zip(
                        cols,
                        [
                            correlation_plot[cat_data[var]],
                            correlation_plot[cont_data[var2]],
                            cor_value,
                        ],
                    )
                ),
                ignore_index=True,
            )
            weighted, unweighted, data = get_msd(
                cont_data[var], cont_data[var2], df, pop_prop_1, 2
            )
            msd_cat_con = msd_cat_con.append(
                dict(
                    zip(
                        final_col,
                        [cat_data[var], cont_data[var2], weighted, unweighted, data],
                    )
                ),
                ignore_index=True,
            )

    return cat_con_cor, cat_con_matrix, msd_cat_con


def cat_cat_corr(cat_data, df, correlation_plot):
    cat_cat_matrix = pd.DataFrame(index=cat_data, columns=cat_data)
    pop_prop_1 = df.response.sum() / len(df)
    final_col = [
        "Cat_var1",
        "Cat_var2",
        "MeanSquareDiff",
        "MeanSquareDiffWeighted",
        "PLot",
    ]
    msd_cat_cat = pd.DataFrame(columns=final_col)
    cols = ["Cat_var1", "Cat_var2", "Correlation"]
    cat_cat_corr = pd.DataFrame(columns=cols)
    for var in range(len(cat_data)):
        for var2 in range(len(cat_data)):
            cramer_value = cat_correlation(df[cat_data[var]], df[cat_data[var2]])
            cat_cat_matrix.loc[cat_data[var]][cat_data[var2]] = cramer_value
            cat_cat_matrix.loc[cat_data[var2]][cat_data[var]] = cramer_value
            cat_cat_corr = cat_cat_corr.append(
                dict(
                    zip(
                        cols,
                        [
                            correlation_plot[cat_data[var]],
                            correlation_plot[cat_data[var2]],
                            cramer_value,
                        ],
                    )
                ),
                ignore_index=True,
            )
            weighted, unweighted, data = get_msd(
                cat_data[var], cat_data[var2], df, pop_prop_1, 2
            )
            msd_cat_cat = msd_cat_cat.append(
                dict(
                    zip(
                        final_col,
                        [cat_data[var], cat_data[var2], weighted, unweighted, data],
                    )
                ),
                ignore_index=True,
            )

    return cat_cat_corr, cat_cat_matrix, msd_cat_cat


def msd_cat_cont(cat_col, cont_col, inp_data):
    bin1 = pd.DataFrame(
        {
            "X1": inp_data[cat_col],
            "X2": inp_data[cont_col],
            "Y": inp_data["response"],
            "Bucket": pd.qcut(inp_data[cont_col], 3),
        }
    )
    bin2 = bin1.groupby(["X1", "Bucket"]).agg({"Y": ["count", "mean"]}).reset_index()
    return bin2


def msd_cont_cont(col1, col2, inp_data):
    bin1 = pd.DataFrame(
        {
            "X1": inp_data[col1],
            "X2": inp_data[col2],
            "Y": inp_data["response"],
            "Bucket1": pd.qcut(inp_data[col1], 3),
            "Bucket2": pd.qcut(inp_data[col2], 3),
        }
    )
    bin2 = (
        bin1.groupby(["Bucket1", "Bucket2"]).agg({"Y": ["count", "mean"]}).reset_index()
    )

    return bin2


def get_msd(col1, col2, inp_data, pop_prop_1, t):
    if t == 3:
        d1_c_c = pd.DataFrame(
            {
                "X1": inp_data[col1],
                "X2": inp_data[col2],
                "Y": inp_data["target"],
            }
        )
        bin2 = d1_c_c.groupby(["X1", "X2"]).agg({"Y": ["count", "mean"]}).reset_index()

    elif t == 2:
        bin2 = msd_cat_cont(col1, col2, inp_data)
    else:
        bin2 = msd_cont_cont(col1, col2, inp_data)

    bin2.columns = [col1, col2, "BinCount", "BinMean"]
    pop_prop = bin2.BinCount / len(inp_data)
    bin2["Mean_sq_diff"] = (bin2["BinMean"] - pop_prop_1) ** 2
    bin2["Mean_sq_diffW"] = bin2.Mean_sq_diff * pop_prop

    # Creating MSD plots
    d_mat = bin2.pivot(index=col1, columns=col2, values="Mean_sq_diffW")
    fig = graph_objects.Figure(data=[graph_objects.Surface(z=d_mat.values)])
    fig.update_layout(
        title=col1 + " " + col2 + " Plot",
        autosize=True,
        scene=dict(xaxis_title=col2, yaxis_title=col1, zaxis_title="target"),
    )

    filename = "midterm/BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    file_n = "midterm/BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    file_name = "<a href=" + file_n + ">Plot Link"
    return bin2["Mean_sq_diff"].sum(), bin2["Mean_sq_diffW"].sum(), file_name


if __name__ == "__main__":
    file = "data.csv"
    response = "diagnosis"
    # file = sys.argv[1]
    # response = sys.argv[2]
    sys.exit(main(file, response))
