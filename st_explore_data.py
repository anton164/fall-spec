import streamlit as st
import os
import pandas as pd
import plotly.express as px
import json

st.header("Explore cached datasets")

data_dir = "./data/tweet_counts"
files = os.listdir(data_dir)

selected_file = os.path.join(
    data_dir,
    st.selectbox("Select a file", options=files)
)
with open(selected_file, "r") as f:
    file_content = json.load(f)
    query_params = file_content["query_params"]
    df_timeseries = pd.DataFrame(file_content["data"])
st.write("### Query Params")
st.write(query_params)
st.write(f"**Total tweet count:** {df_timeseries.tweet_count.sum():,}")
st.write(px.line(df_timeseries, x="time_end", y="tweet_count"))
