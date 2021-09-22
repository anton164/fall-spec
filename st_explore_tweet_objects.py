import streamlit as st
import os
import json
import numpy as np
import pandas as pd

st.header("Explore tweet object datasets")

data_dir = "./data/tweet_objects"
handles = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
selected_handle = st.selectbox("Select a handle", options=handles)

timespans = os.listdir(os.path.join(data_dir, selected_handle))
selected_timespan = st.selectbox("Select a timespan", options=timespans)

with open(os.path.join(data_dir, selected_handle, selected_timespan), 'r') as f:
    data = json.load(f)

st.subheader("Query params")
st.write(data["query_params"])

earliest_tweet_date = None
latest_tweet_date = None
for tweet in data["tweets"]["data"]:
    created_date = pd.to_datetime(tweet["created_at"])
    if earliest_tweet_date is None or created_date < earliest_tweet_date:
        earliest_tweet_date = created_date
    if latest_tweet_date is None or created_date > latest_tweet_date:
        latest_tweet_date = created_date

st.write(
f"""
**Earliest tweet date:** {earliest_tweet_date}    
**Latest tweet date:** {latest_tweet_date}  
**Includes NER tags:** {data["include_ner_tags"]}  
**Number of tweets:** {len(data["tweets"]["data"])}  
**Number of user objects:** {len(data["tweets"]["includes"]["users"])}
"""
)

st.subheader("Tweet sample")
sampled_tweets = np.random.choice(data["tweets"]["data"])
st.write(sampled_tweets)