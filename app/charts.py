import plotly.express as px
import pandas as pd


def revenue_chart(df):

    fig = px.line(
        df,
        x="InvoiceDate",
        y="Revenue",
        title="Revenue Trend"
    )

    return fig


def country_chart(df):

    grouped = (
        df.groupby("Country")["Revenue"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        grouped.sort_values(
            "Revenue",
            ascending=False
        ).head(10),
        x="Country",
        y="Revenue",
        title="Top Countries"
    )

    return fig