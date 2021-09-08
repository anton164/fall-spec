import streamlit as st
import os
import pandas as pd
import plotly.express as px
from dateutil.relativedelta import relativedelta
import json
import urllib.parse


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

st.subheader("Explore tweets on Twitter for this query")
date = st.date_input("Select date")
start_date = date - relativedelta(days=1)
end_date = date + relativedelta(days=1)
min_retweets = st.number_input("Minimum retweets", value=0)
query_str = f"({query_params['query']}) AND since:{start_date} AND until:{end_date} AND min_retweets:{min_retweets}"
st.write(f"Query: `{query_str}`")
st.write(f"[Open tweets in Twitter](https://twitter.com/search?q={urllib.parse.quote_plus(query_str)}&src=typed_query&f=top)")