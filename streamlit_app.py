import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Used Phones & Tablets Price Analysis", page_icon="📱", layout="wide"
)

# Title and Description
st.title("📱 Used Phones & Tablets Price Analysis")
st.write(
    "This web app visualizes data from the "
    "[Used Phones & Tablets Prices dataset](https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data). "
    "You can explore price trends based on brand, RAM, internal storage, and other features."
)

st.write(
    "🚀 Built with Streamlit | Data Source: Kaggle | Developed by **ECE143_WI25_Team3** 🌊"
)


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("./data/used_device_data.csv")


df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Options")

# Device Brand Selection
selected_brands = st.sidebar.multiselect(
    "Select Device Brand", df["device_brand"].unique()
)

# Operating System Selection
selected_os = st.sidebar.multiselect("Select OS", df["os"].unique())

# 4G & 5G Support
selected_4g = st.sidebar.radio("4G Support", ["All", "Yes", "No"], index=0)
selected_5g = st.sidebar.radio("5G Support", ["All", "Yes", "No"], index=0)

# Internal Memory & RAM Filters
selected_memory = st.sidebar.slider(
    "Minimum Internal Memory (GB)",
    int(df["internal_memory"].min()),
    int(df["internal_memory"].max()),
    int(df["internal_memory"].min()),
)
selected_ram = st.sidebar.slider(
    "Minimum RAM (GB)", int(df["ram"].min()), int(df["ram"].max()), int(df["ram"].min())
)

# Release Year Range
selected_years = st.sidebar.slider(
    "Select Release Year Range",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (2015, 2021),
)

# --- Filtering Data ---
filtered_df = df.copy()

if selected_brands:
    filtered_df = filtered_df[filtered_df["device_brand"].isin(selected_brands)]

if selected_os:
    filtered_df = filtered_df[filtered_df["os"].isin(selected_os)]

if selected_4g != "All":
    filtered_df = filtered_df[
        filtered_df["4g"] == ("yes" if selected_4g == "Yes" else "no")
    ]

if selected_5g != "All":
    filtered_df = filtered_df[
        filtered_df["5g"] == ("yes" if selected_5g == "Yes" else "no")
    ]

filtered_df = filtered_df[
    (filtered_df["internal_memory"] >= selected_memory)
    & (filtered_df["ram"] >= selected_ram)
    & (filtered_df["release_year"].between(*selected_years))
]

# Create two columns
col_a, col_b = st.columns(2)

# --- Brand Distribution in the First Column ---
with col_a:
    st.subheader("📊 Brand Distribution")
    brand_counts = filtered_df["device_brand"].value_counts().reset_index()
    brand_counts.columns = ["device_brand", "count"]

    brand_chart = (
        alt.Chart(brand_counts)
        .mark_bar()
        .encode(
            x=alt.X("device_brand:N", title="Device Brand", sort="-y"),
            y=alt.Y("count:Q", title="Count"),
            color="device_brand:N",
            tooltip=["device_brand", "count"],
        )
        .properties(height=400)
    )

    st.altair_chart(brand_chart, use_container_width=True)

# --- Average Used Price by Brand in the Second Column ---
with col_b:
    st.subheader("💰 Average Used Price by Brand")
    avg_price_by_brand = (
        filtered_df.groupby("device_brand")["normalized_used_price"]
        .mean()
        .reset_index()
        .sort_values(by="normalized_used_price", ascending=False)
    )

    price_chart = (
        alt.Chart(avg_price_by_brand)
        .mark_bar()
        .encode(
            x=alt.X("device_brand:N", title="Device Brand", sort="-y"),
            y=alt.Y("normalized_used_price:Q", title="Avg. Used Price"),
            color="device_brand:N",
            tooltip=["device_brand", "normalized_used_price"],
        )
        .properties(height=400)
    )

    st.altair_chart(price_chart, use_container_width=True)

# Add a separator line
st.markdown("---")


# --- Layout: Two-Column View ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Filtered Device Prices")
    st.dataframe(
        filtered_df[
            [
                "device_brand",
                "os",
                "release_year",
                "internal_memory",
                "ram",
                "battery",
                "normalized_used_price",
            ]
        ],
        use_container_width=True,
    )

with col2:
    st.subheader("📈 Price Distribution by Brand")
    price_chart = (
        alt.Chart(filtered_df)
        .mark_point()
        .encode(
            x=alt.X("release_year:N", title="Release Year"),
            y=alt.Y("normalized_used_price:Q", title="Normalized Used Price"),
            color="device_brand:N",
            tooltip=["device_brand", "release_year", "normalized_used_price"],
        )
        .properties(height=400)
    )
    st.altair_chart(price_chart, use_container_width=True)

# --- New vs. Used Price Analysis ---
st.subheader("💰 New vs. Used Price Analysis")

# Scatter Plot: New Price vs. Used Price
price_comparison_chart = (
    alt.Chart(filtered_df)
    .mark_circle(size=80)
    .encode(
        x=alt.X("normalized_new_price:Q", title="Normalized New Price"),
        y=alt.Y("normalized_used_price:Q", title="Normalized Used Price"),
        color="device_brand:N",
        tooltip=["device_brand", "normalized_new_price", "normalized_used_price"],
    )
    .properties(height=400)
)

st.altair_chart(price_comparison_chart, use_container_width=True)

# --- Additional Fancy Visualizations ---
st.subheader("📊 Price vs. Internal Storage & RAM")

col3, col4 = st.columns(2)

# Internal Memory vs. Price
with col3:
    st.write("📌 Price vs. Internal Memory")
    memory_chart = (
        alt.Chart(filtered_df)
        .mark_circle(size=80)
        .encode(
            x="internal_memory:Q",
            y="normalized_used_price:Q",
            color="device_brand:N",
            tooltip=["device_brand", "internal_memory", "normalized_used_price"],
        )
        .properties(height=350)
    )
    st.altair_chart(memory_chart, use_container_width=True)

# RAM vs. Price
with col4:
    st.write("📌 Price vs. RAM")
    ram_chart = (
        alt.Chart(filtered_df)
        .mark_circle(size=80)
        .encode(
            x="ram:Q",
            y="normalized_used_price:Q",
            color="device_brand:N",
            tooltip=["device_brand", "ram", "normalized_used_price"],
        )
        .properties(height=350)
    )
    st.altair_chart(ram_chart, use_container_width=True)


col5, col6 = st.columns([2, 1])

# --- Summary Statistics in the Second Column ---
with col5:
    st.subheader("📌 Summary Statistics of Filtered Dataset")
    st.write(filtered_df.describe())
# --- Correlation Heatmap in the First Column ---
with col6:
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        filtered_df[
            [
                "screen_size",
                "internal_memory",
                "ram",
                "battery",
                "weight",
                "days_used",
                "normalized_used_price",
            ]
        ].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    st.pyplot(fig)


st.markdown("---")
