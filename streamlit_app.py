import altair as alt
import pandas as pd
import streamlit as st


# Set page title
st.set_page_config(page_title="Used Devices Price Analysis", page_icon="📱")
st.title("📱 Used Phones & Tablets Price Analysis")

st.write(
    """
    This app visualizes data from the 
    [Used Phones & Tablets Prices dataset](https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data).
    You can explore price trends based on brand, RAM, internal storage, and other features.
    """
)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./data/used_device_data.csv")
    return df


df = load_data()

# Brand selection
brands = st.multiselect(
    "Select Device Brand",
    df["device_brand"].unique(),
    default=df["device_brand"].unique()[:3],
)

# Filter dataset
df_filtered = df[(df["device_brand"].isin(brands))]

st.subheader("📊 Filtered Device Prices")
st.dataframe(
    df_filtered[["device_brand", "os", "release_year", "normalized_used_price"]]
)

# Price visualization
st.subheader("📈 Price Distribution by Brand")

chart = (
    alt.Chart(df_filtered)
    .mark_boxplot()
    .encode(
        x="device_brand:N",
        y="normalized_used_price:Q",
        color="device_brand:N",
        tooltip=["device_brand", "release_year", "normalized_used_price"],
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)

# Additional Distribution: Device Usage Days
st.subheader("📅 Device Usage Duration Distribution")

hist_days_used = (
    alt.Chart(df_filtered)
    .mark_bar()
    .encode(
        x=alt.X("days_used:Q", bin=True, title="Days Used"),
        y="count()",
        tooltip=["days_used"],
    )
    .properties(height=300)
)

st.altair_chart(hist_days_used, use_container_width=True)

# Release Year Trend
st.subheader("📅 Device Release Year Trend")


hist_release_year = (
    alt.Chart(df_filtered)
    .mark_bar()
    .encode(
        x=alt.X("release_year:O", title="Release Year"),
        y="count()",
        tooltip=["release_year"],
    )
    .properties(height=300)
)

st.altair_chart(hist_release_year, use_container_width=True)


# from pydantic_settings import BaseSettings

# import pandas_profiling as pp
# from streamlit_pandas_profiling import st_profile_report

# # df = pd.read_csv("./data/used_device_data.csv")
# pr = df.profile_report()

# st_profile_report(pr)
