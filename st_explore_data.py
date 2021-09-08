import streamlit as st
import os
import pandas as pd
import plotly.express as px

st.header("Explore cached datasets")

data_dir = "./data"
files = os.listdir(data_dir)

selected_file = os.path.join(
    data_dir,
    st.selectbox("Select a file", options=files)
)

df_timeseries = pd.read_json(selected_file, orient="records")
st.write(f"**Total tweet count:** {df_timeseries.tweet_count.sum():,}")
st.write(px.line(df_timeseries, x="time_end", y="tweet_count"))