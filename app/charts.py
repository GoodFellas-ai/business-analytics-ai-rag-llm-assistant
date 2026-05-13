import plotly.express as px
import pandas as pd


def revenue_chart(df):

    if df is None or df.empty:
        return px.line(title="Upload CSV to see revenue trend")

    if "InvoiceDate" not in df.columns or "Revenue" not in df.columns:
        return px.line(title="Missing required columns")

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "Revenue"])

    grouped = df.groupby("InvoiceDate")["Revenue"].sum().reset_index()

    return px.line(
        grouped,
        x="InvoiceDate",
        y="Revenue",
        title="Revenue Trend"
    )


def country_chart(df):

    if df is None or df.empty:
        return px.bar(title="Upload CSV to see country data")

    if "Country" not in df.columns or "Revenue" not in df.columns:
        return px.bar(title="Missing required columns")

    grouped = df.groupby("Country")["Revenue"].sum().reset_index()

    return px.bar(
        grouped.sort_values("Revenue", ascending=False).head(10),
        x="Country",
        y="Revenue",
        title="Top Countries"
    )
