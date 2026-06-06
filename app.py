import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("📊 AI Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Data Cleaning")

    missing = df.isnull().sum()
    st.write("Missing Values")
    st.write(missing)

    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    st.success("Cleaning Completed")

    st.sidebar.header("Filters")

    column_filter = st.sidebar.selectbox(
        "Choose Column",
        df.columns
    )

    unique_values = df[column_filter].dropna().unique()

    selected = st.sidebar.multiselect(
        "Filter Values",
        unique_values,
        default=unique_values[:5]
    )

    filtered_df = df[
        df[column_filter].isin(selected)
    ]

    st.subheader("KPIs")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        filtered_df.shape[0]
    )

    c2.metric(
        "Columns",
        filtered_df.shape[1]
    )

    c3.metric(
        "Missing Values",
        filtered_df.isnull().sum().sum()
    )

    st.subheader("Visualizations")

    num_cols = filtered_df.select_dtypes(
        include=np.number
    ).columns

    cat_cols = filtered_df.select_dtypes(
        exclude=np.number
    ).columns

    if len(num_cols) > 0:

        fig1 = px.histogram(
            filtered_df,
            x=num_cols[0]
        )
        st.plotly_chart(fig1)

        fig2 = px.box(
            filtered_df,
            y=num_cols[0]
        )
        st.plotly_chart(fig2)

        if len(num_cols) > 1:

            fig3 = px.scatter(
                filtered_df,
                x=num_cols[0],
                y=num_cols[1]
            )
            st.plotly_chart(fig3)

            corr = filtered_df[num_cols].corr()

            fig4 = px.imshow(
                corr,
                text_auto=True
            )
            st.plotly_chart(fig4)

    if len(cat_cols) > 0:

        counts = filtered_df[
            cat_cols[0]
        ].value_counts().head(10)

        fig5 = px.bar(
            x=counts.index,
            y=counts.values
        )

        st.plotly_chart(fig5)

else:
    st.info("Upload CSV file")