from merlion.models.anomaly.forecast_based.prophet import ProphetDetector, ProphetDetectorConfig
from merlion.post_process.threshold import Threshold
import streamlit as st
import pandas as pd
import pickle

from utils.label_spikes import detect_anomalies, timeseries_from_query_counts

def render_celeb_dataset():
    st.header("Explore celebrity dataset")

    df_celeb = pd.read_csv("./data/celebrity/enhanced_1k_celebrities_with_spikes.csv").set_index("twitter_username")
    df_celeb["created_at"] = pd.to_datetime(df_celeb.created_at)
    with open("./data/celebrity/tweet_count_data_by_user", "rb") as f:
        cached_tweet_count_data_by_user = pickle.load(f)

    st.write(f"Number of handles: {df_celeb.shape[0]}")
    st.write(df_celeb)

    col1, col2 = st.columns(2)

    max_query_count = col1.number_input("Max number of tweets in timeseries", 100000)
    spikes_above_2 = col2.number_input("Number of spikes with anom score above 2", value=0)

    df_celeb_filtered = df_celeb[
        (df_celeb.query_tweet_count <= max_query_count) &
        (df_celeb.spikes_above_2 >= spikes_above_2)
    ].drop(["name", "created_at"], axis=1)

    st.write(f"Number of filtered handles: {df_celeb_filtered.shape[0]}")

    st.write(df_celeb_filtered)

    selected_handle = st.selectbox("Select handle", options=df_celeb_filtered.index)
    st.write(selected_handle)
    user = df_celeb_filtered.loc[selected_handle]
    st.write(user)
    if st.checkbox("Detect Anomalies"):

        prophet_config = ProphetDetectorConfig(
            threshold=Threshold(alm_threshold=0.5),
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            add_seasonality=True,  
            uncertainty_samples=1000
        )

        detected_spikes_by_user = {}
        df_timeseries = timeseries_from_query_counts(cached_tweet_count_data_by_user[user.name])
                
        df_scores, df_labels, fig = anomaly_scores = detect_anomalies(
            df_timeseries, 
            "tweet_count",
            model=ProphetDetector(prophet_config),
            include_plot=True
        )
        st.write(fig)

if __name__ == "__main__":
    render_celeb_dataset()