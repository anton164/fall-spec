from utils.mstream import load_mstream_predictions
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
    mstream_scores_file = f"./MStream/data/{dataset_name}_score.txt"
    mstream_labels_file = f"./MStream/data/{dataset_name}_predictions.txt"
    mstream_decomposed_scores_file = f"./MStream/data/{dataset_name}_decomposed.txt"
    mstream_decomposed_p_scores_file = f"./MStream/data/{dataset_name}_decomposed_percentage.txt"
    columns_names_file = f"./MStream/data/{dataset_name}_columns.txt"

    try:
        df_mstream_input = pd.read_pickle(f"./MStream/data/{dataset_name}_data.pickle")
        df_tweets_with_mstream_output = load_mstream_predictions(
            df_tweets,
            mstream_scores_file,
            mstream_labels_file,
            mstream_decomposed_scores_file,
            mstream_decomposed_p_scores_file,
            columns_names_file
        )
        df_mstream_input["mstream_anomaly_score"] = df_mstream_input.apply(
            lambda t: df_tweets_with_mstream_output.loc[t.name].mstream_anomaly_score,
            axis=1
        )
        if "text_score" in df_tweets_with_mstream_output.columns:
            df_mstream_input["mstream_text_score"] = df_mstream_input.apply(
                lambda t: df_tweets_with_mstream_output.loc[t.name].text_score,
                axis=1
            )
        if "hashtag_score" in df_tweets_with_mstream_output.columns:
            df_mstream_input["mstream_hashtag_score"] = df_mstream_input.apply(
                lambda t: df_tweets_with_mstream_output.loc[t.name].hashtag_score,
                axis=1
            )
    except Exception as e:
        st.error(f"Failed to load MStream output for {dataset_name}")
        raise e

    fig = go.Figure()


    show_mstream_input = st.button("Show MStream input")
    if show_mstream_input:
        st.subheader("MStream input")
        st.write(df_mstream_input)

        df_mstream_input["created_at"] = pd.to_datetime(df_mstream_input["created_at"])
        unique_tokens = set()
        def unique_tokens_over_time(text):
            for token in text:
                unique_tokens.add(token)            
            return len(unique_tokens)
        df_unique_tokens = df_mstream_input.groupby(
            df_mstream_input.created_at.dt.ceil(time_bucket_size)
        ).agg(
            unique_tokens=(
                'text', 
                lambda text_col: text_col.apply(lambda text: unique_tokens_over_time(text)).max()
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
    for i, val in enumerate(df_tweets.columns):
        if val in read_columns(columns_names_file) and val != 'record_score':
            hovertext = df_tweets[val.split('_')[0]].apply(lambda x: ','.join(map(str, x))).tolist()
            fig.add_trace(
                go.Scatter(
                    x=df_tweets.created_at,
                    y=df_tweets[val],
                    mode='lines',
                    name=" ".join(val.split('_')),
                    opacity=0.5,
                    hovertext=hovertext
                )
            )
    selected_points = plotly_events(fig)
    #st.write(fig)

if __name__ == "__main__":
    render_mstream_results()