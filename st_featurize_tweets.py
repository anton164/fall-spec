from utils.mstream import load_mstream_predictions
from utils.st_utils import st_select_file
from utils.dataset import count_array_column, load_tweet_dataset
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils.dtypes import tweet_dtypes
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

st.header("Featurize dataset")

data_dir = "./data/labeled_datasets"
selected_dataset = st_select_file(
    "Select dataset",
    "./data/labeled_datasets",
    ".json"
)

df_tweets = load_tweet_dataset(selected_dataset)
merlin_anomaly_threshold = st.number_input("Merlin anomaly threshold", value=1)
df_tweets["is_anomaly"] = df_tweets["merlion_anomaly_total_count"].apply(lambda x: x > merlin_anomaly_threshold)

df_retweets = df_tweets[pd.notna(df_tweets.retweeted)]
df_quotes = df_tweets[pd.notna(df_tweets.quoted)]
df_replies = df_tweets[pd.notna(df_tweets.replied_to)]
df_og_tweets = df_tweets[
    pd.isna(df_tweets.retweeted) & 
    pd.isna(df_tweets.quoted) & 
    pd.isna(df_tweets.replied_to)
]

tweet_ids = set(df_tweets.id)
retweeted_ids = set(df_tweets.retweeted)
quoted_ids = set(df_tweets.quoted)
replied_ids = set(df_tweets.replied_to)

st.write(
f"""
**Earliest tweet date:** {df_tweets.created_at.min()}    
**Latest tweet date:** {df_tweets.created_at.max()}  
**Number of rows:** {len(df_tweets)}  
**Number of original tweets:** {len(df_og_tweets)}  
**Number of retweets:** {len(df_retweets)} (retweeted {len(retweeted_ids)} unique tweets)  
**Number of quotes:** {len(df_quotes)} (quoted {len(quoted_ids)} unique tweets)  
**Number of replies:** {len(df_replies)} (replied to {len(replied_ids)} unique tweets)  
**Number of user objects:** {df_tweets.user_id.nunique()}
"""
)

df_tweets["retweeted_missing"] = df_tweets.retweeted.apply(lambda x: x not in tweet_ids if x is not None else False)
df_tweets["quoted_missing"] = df_tweets.quoted.apply(lambda x: x not in tweet_ids if x is not None else False)
df_tweets["replied_to_missing"] = df_tweets.replied_to.apply(lambda x: x not in tweet_ids if x is not None else False)

missing_retweeted_ids = len(retweeted_ids - tweet_ids)
missing_quoted_ids = len(quoted_ids - tweet_ids)
missing_replied_to_ids = len(replied_ids - tweet_ids)

with st.expander("Dataset completeness"):
    st.write(
        f"""
            **Retweets:** {missing_retweeted_ids}/{len(retweeted_ids)} unique tweets missing in {df_tweets["retweeted_missing"].sum()/len(df_retweets):.1%} retweets    
            **Quotes:** {missing_quoted_ids}/{len(quoted_ids)} unique tweets missing in {df_tweets["quoted_missing"].sum()/len(df_quotes):.1%} quote tweets  
            **Replies:** {missing_replied_to_ids}/{len(replied_ids)} unique tweets missing in {df_tweets["replied_to_missing"].sum()/len(df_replies):.1%} replies
        """
    )

with st.expander("Top tweets, hashtags & mentions"):
    st.subheader("Top 5 tweets by retweets")
    st.write(df_tweets.nlargest(5, "retweet_count"))

    col1, col2  = st.columns(2)
    col1.subheader("Top hashtags")
    df_top_hashtags = count_array_column(df_tweets["hashtags"])
    col1.write(
        df_top_hashtags[:10]
    )
    col2.subheader("Top mentions")
    df_top_mentions = count_array_column(df_tweets["mentions"])
    col2.write(
        df_top_mentions[:10]
    )

st.header("Explore time series of features")

def count_col_occurrence(df_col, value):
    return df_col.apply(lambda values: value in values).sum()

time_bucket_size = st.text_input("Time bucket size", value="30Min") 

df_timeseries = df_tweets.groupby(df_tweets.created_at.dt.ceil(time_bucket_size)).agg(
    total_count=('id', 'count'), 
    hashtag_count=('hashtags', lambda hashtag_col: hashtag_col.apply(lambda x: len(x)).sum()),
    is_anomaly=('is_anomaly', lambda x: x.any()),
    merlion_count_anomaly=('merlion_anomaly_total_count', lambda x: x.max()),
    merlion_hashtag1_anomaly=('merlion_anomaly_top1_hashtag_count', lambda x: x.max()),
    merlion_hashtag2_anomaly=('merlion_anomaly_top2_hashtag_count', lambda x: x.max()),
    merlion_hashtag3_anomaly=('merlion_anomaly_top3_hashtag_count', lambda x: x.max()),
    retweet_count=('retweeted', lambda x: pd.notna(x).sum()),
    quote_count=('quoted', lambda x: pd.notna(x).sum()),
    replied_to_count=('replied_to', lambda x: pd.notna(x).sum()),
    top1_hashtag_count=(
        'hashtags', 
        lambda x: count_col_occurrence(x, df_top_hashtags.iloc[0].value)
    ),
    top2_hashtag_count=(
        'hashtags', 
        lambda x: count_col_occurrence(x, df_top_hashtags.iloc[1].value)
    ),
    top3_hashtag_count=(
        'hashtags', 
        lambda x: count_col_occurrence(x, df_top_hashtags.iloc[2].value)
    ),
    top1_mention_count=(
        'mentions', 
        lambda x: count_col_occurrence(x, df_top_mentions.iloc[0].value)
    ),
    top2_mention_count=(
        'mentions', 
        lambda x: count_col_occurrence(x, df_top_mentions.iloc[1].value)
    ),
    top3_mention_count=(
        'mentions', 
        lambda x: count_col_occurrence(x, df_top_mentions.iloc[2].value)
    )
)

df_timeseries["tweet_count"] = df_timeseries.apply(
    lambda x: x.total_count - (x.retweet_count + x.replied_to_count + x.quote_count),
    axis=1
)

# Rename hashtag & mention columns
df_timeseries = df_timeseries.rename(
    lambda col: col if "hashtag" not in col or col == "hashtag_count" or col.startswith("merlion") else "#" + df_top_hashtags.iloc[
        int(next(filter(str.isdigit, col))) - 1   
    ].value,
    axis=1
)

df_timeseries = df_timeseries.rename(
    lambda col: col if "mention" not in col else "@" + df_top_mentions.iloc[
        int(next(filter(str.isdigit, col))) - 1   
    ].value,
    axis=1
)

# anomaly based on merlion
df_timeseries["is_anomaly_hashtag1"] = df_timeseries["merlion_hashtag1_anomaly"].apply(lambda x: x > merlin_anomaly_threshold)
df_timeseries["is_anomaly_hashtag2"] = df_timeseries["merlion_hashtag2_anomaly"].apply(lambda x: x > merlin_anomaly_threshold)
df_timeseries["is_anomaly_hashtag3"] = df_timeseries["merlion_hashtag3_anomaly"].apply(lambda x: x > merlin_anomaly_threshold)

st.write(df_timeseries.head())

selected_columns = st.multiselect(
    "Select columns",
    options=df_timeseries.columns,
    default=[
        "hashtag_count",
    ]
)
fig = go.Figure()
for col in selected_columns:
    fig.add_trace(
        go.Scatter(
            x=df_timeseries.index,
            y=df_timeseries[col],
            mode='lines',
            name=col
        )
    )
    if col == "total_count":
        fig.add_trace(
            go.Scatter(
                x=df_timeseries[df_timeseries.is_anomaly].index,
                y=df_timeseries[df_timeseries.is_anomaly]["total_count"],
                mode='markers',
                name="total count anomaly"
            )
        )
    elif col == "#" + df_top_hashtags.iloc[0].value:
        fig.add_trace(
            go.Scatter(
                x=df_timeseries[df_timeseries.is_anomaly_hashtag1].index,
                y=df_timeseries[df_timeseries.is_anomaly_hashtag1]["#" + df_top_hashtags.iloc[0].value],
                mode='markers',
                name="hashtag 1 anomaly"
            )
        )
    elif col == "#" + df_top_hashtags.iloc[1].value:
        fig.add_trace(
            go.Scatter(
                x=df_timeseries[df_timeseries.is_anomaly_hashtag2].index,
                y=df_timeseries[df_timeseries.is_anomaly_hashtag2]["#" + df_top_hashtags.iloc[1].value],
                mode='markers',
                name="hashtag 2 anomaly"
            )
        )
    elif col == "#" + df_top_hashtags.iloc[2].value:
        fig.add_trace(
            go.Scatter(
                x=df_timeseries[df_timeseries.is_anomaly_hashtag3].index,
                y=df_timeseries[df_timeseries.is_anomaly_hashtag3]["#" + df_top_hashtags.iloc[2].value],
                mode='markers',
                name="hashtag 3 anomaly"
            )
        )

st.write(fig)

st.header("Explore MStream Results")
mstream_scores_file = st_select_file(
    "Select generated scores", 
    data_dir="./MStream/data",
    extension="_score.txt"
)
mstream_labels_file = st_select_file(
    "Select generated labels", 
    data_dir="./MStream/data",
    extension="_predictions.txt"
)
mstream_decomposed_scores_file = st_select_file(
    "Select decomposed scores", 
    data_dir="./MStream/data",
    extension="_decomposed.txt"
)
df_tweets_with_mstream_output = load_mstream_predictions(
    df_tweets,
    mstream_scores_file,
    mstream_labels_file,
    mstream_decomposed_scores_file
)

fig = go.Figure()

st.write(df_tweets.head())

st.write(f"""
**Precision:** {precision_score(df_tweets.is_anomaly, df_tweets.mstream_is_anomaly):.2%}  
**Recall:** {recall_score(df_tweets.is_anomaly, df_tweets.mstream_is_anomaly):.2%}  
**F1_score:** {f1_score(df_tweets.is_anomaly, df_tweets.mstream_is_anomaly):.2%}
""")

max_mstream_score = df_tweets.mstream_anomaly_score.max()

fig.add_trace(
    go.Scatter(
        x=df_tweets.created_at,
        y=df_tweets.is_anomaly.astype(int)*max_mstream_score + 20,
        mode='lines',
        name="True labels",
        opacity=0.5
    )
)

fig.add_trace(
    go.Scatter(
        x=df_tweets.created_at,
        y=df_tweets.mstream_is_anomaly.astype(int)*max_mstream_score,
        mode='lines',
        name="Predicted labels",
        opacity=0.5
    )
)

fig.add_trace(
    go.Scatter(
        x=df_tweets.created_at,
        y=df_tweets.mstream_anomaly_score,
        mode='lines',
        name="Anomaly score",
        opacity=0.5
    )
)
for i, val in enumerate(df_tweets.columns):
    if 'mstream_decomposed_anomaly_score' in val:
        fig.add_trace(
            go.Scatter(
                x=df_tweets.created_at,
                y=df_tweets[val],
                mode='lines',
                name=val,
                opacity=0.5
            )
        )

st.write(fig)