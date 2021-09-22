import streamlit as st
import os
import pandas as pd
import plotly.express as px
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone
import json
import urllib.parse
from api.twitter import fetch_historical_tweets, HistoricalTweetsParams


st.header("Explore cached datasets")

data_dir = "./data/tweet_counts"
files = os.listdir(data_dir)
selected_filename = st.selectbox("Select a file", options=files)
dataset_name = selected_filename.replace(".json", "")
resampling_period = st.text_input("Resample dataframe", value="24H")
with open(os.path.join(data_dir, selected_filename), "r") as f:
    file_content = json.load(f)
    query_params = file_content["query_params"]
    df_timeseries = pd.DataFrame(file_content["data"])
    df_timeseries["time_end"] = pd.to_datetime(df_timeseries.time_end)
    df_timeseries = df_timeseries.set_index("time_end").resample(resampling_period).sum()

st.write("### Query Params")
st.write(query_params)
st.write(f"**Total tweet count:** {df_timeseries.tweet_count.sum():,}")
st.write(px.line(df_timeseries, y="tweet_count"))

explore_type = st.radio("Explore tweets", options=[
    "Through Twitter UI", 
    "Through Historical Tweet API"
])

if explore_type == "Through Twitter UI":
    st.subheader("Explore tweets on Twitter UI for this query")
    date = st.date_input("Select date")
    start_date = date - relativedelta(days=1)
    end_date = date + relativedelta(days=1)
    min_retweets = st.number_input("Minimum retweets", value=0)
    query_str = f"({query_params['query']}) AND since:{start_date} AND until:{end_date} AND min_retweets:{min_retweets}"
    st.write(f"Query: `{query_str}`")
    st.write(f"[Open tweets in Twitter](https://twitter.com/search?q={urllib.parse.quote_plus(query_str)}&src=typed_query&f=top)")
else:
    st.subheader("Explore historical tweets on Twitter for this query")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", value=datetime(2020, 9, 1))
    end_date = col2.date_input("End date", value=datetime(2020, 9, 1))
    include_ner_tags = st.checkbox("Include Context Annotations (NER Tags etc. max 100 per page & slows down fetch)")
    n_pages = st.number_input("Number of pages (~100/500 tweets per page)", value=1)

    query_params: HistoricalTweetsParams = {
        "query": query_params["query"],
        "start_time": datetime.combine(
            start_date,
            datetime.min.time()
        ).replace(tzinfo=timezone.utc).isoformat(),
        "end_time": datetime.combine(
            end_date,
            datetime.max.time()
        ).replace(tzinfo=timezone.utc).isoformat()
    }

    save_to_file = st.checkbox("Save tweets to data/tweet_objects")
    overwrite_file = st.checkbox("Overwrite previous data/tweet_objects file if it exists")
    if st.button("Fetch tweets"):
        tweets = fetch_historical_tweets(
            query_params,
            include_ner_tags,
            n_pages,
        )

        with st.expander("Show response (might be slow)"):
            st.write(tweets)
        
        if save_to_file:
            dir_name = f"data/tweet_objects/{dataset_name}"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            out_file = f"{dir_name}/{start_date}-{end_date}{('-with-context' if include_ner_tags else '')}.json"
            if os.path.exists(out_file) and not overwrite_file:
                st.error(f"File {out_file} already exists, delete it first")
            else:
                with open(out_file, "w") as f:
                    json.dump({
                        "query_params": query_params,
                        "include_ner_tags": include_ner_tags,
                        "n_pages": n_pages,
                        "tweets": tweets
                    }, f)
                    st.success(f"Wrote to {out_file}!")