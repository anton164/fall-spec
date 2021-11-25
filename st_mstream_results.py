from utils.mstream import load_mstream_predictions, load_mstream_results_for_dataset
from utils.st_utils import st_select_file
from utils.dataset import count_array_column, load_tweet_dataset, read_columns
import streamlit as st
from streamlit_plotly_events import plotly_events
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils.dtypes import tweet_dtypes
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def render_mstream_results():
    time_bucket_size = st.text_input("Time bucket size", value="30Min") 
    data_dir = "./data/labeled_datasets"
    selected_dataset = st_select_file(
        "Select dataset",
        data_dir,
        ".json"
    )
    dataset_name = selected_dataset.replace(".json", "").replace(data_dir + "/", "")
    df_tweets = load_tweet_dataset(selected_dataset)

    st.header("Explore MStream Results")

    df_mstream_input, score_columns = load_mstream_results_for_dataset(
        dataset_name,
        df_tweets
    )

    fig = go.Figure()


    show_mstream_input = st.button("Show MStream input")
    if show_mstream_input:
        st.subheader("MStream input")
        st.write(df_mstream_input)

        df_mstream_input["created_at"] = pd.to_datetime(df_mstream_input["created_at"])
        unique_tokens = set()
        def unique_tokens_over_time(tokens):
            for token in tokens:
                unique_tokens.add(token)            
            return len(unique_tokens)
        df_unique_tokens = df_mstream_input.groupby(
            df_mstream_input.created_at.dt.ceil(time_bucket_size)
        ).agg(
            unique_tokens=(
                'tokens', 
                lambda token_col: token_col.apply(lambda text: unique_tokens_over_time(text)).max()
            ),  
        )
        st.write(px.line(
            df_unique_tokens.unique_tokens,
            title="Number of unique tokens over time"
        ))

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
    fig.add_trace(
        go.Scatter(
            x=df_tweets.created_at,
            y=df_tweets.record_score,
            mode='lines',
            name="record_score",
            opacity=0.5
        )
    )
    for i, val in enumerate(df_tweets.columns):
        if val in score_columns and val not in ['record_score']:
            #hovertext = df_tweets[val.split('_')[0]].apply(lambda x: ','.join(map(str, x))).tolist()
            fig.add_trace(
                go.Scatter(
                    x=df_tweets.created_at,
                    y=df_tweets[val],
                    mode='lines',
                    name=" ".join(val.split('_')),
                    opacity=0.5,
             #       hovertext=hovertext
                )
            )
    selected_points = plotly_events(fig)
    st.write(fig)

    st.subheader("Top Anomalies")
    present_score_columns = [col for col in score_columns if col in df_mstream_input.columns]
    selected_score_column = st.selectbox("By score", options=present_score_columns)
    n_tweets = st.number_input("Number of tweets to show", value=100)
    df_top_anoms = df_mstream_input.nlargest(n_tweets, selected_score_column)
    st.write(
        df_top_anoms
    )

    st.write(f"Percentage of retweets in top {n_tweets}: {df_top_anoms.is_retweet.sum() / df_top_anoms.shape[0]:%}")
    st.write(f"Percentage of retweets in dataset: {df_mstream_input.is_retweet.sum() / df_mstream_input.shape[0]:%}")

    st.write(f"Average token length in top {n_tweets}: {df_top_anoms.token_length.mean()}")
    if "token_length" in df_mstream_input:
        st.write(f"Average token length in dataset: {df_mstream_input.token_length.mean()}")


if __name__ == "__main__":
    render_mstream_results()