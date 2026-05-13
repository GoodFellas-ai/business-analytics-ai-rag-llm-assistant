import plotly.express as px
import pandas as pd


def revenue_chart(df):

    df = df.copy()

    # SAFE DATE FIX
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df = df.dropna(subset=["InvoiceDate", "Revenue"])

    df_grouped = df.groupby("InvoiceDate")["Revenue"].sum().reset_index()

    fig = px.line(
        df_grouped,
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
        grouped.sort_values("Revenue", ascending=False).head(10),
        x="Country",
        y="Revenue",
        title="Top Countries"
    )

    return fig
