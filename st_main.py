import streamlit as st
from st_bucket import render_buckets
from st_celeb_dataset import render_celeb_dataset
from st_explore_tweet_counts import render_explore_tweet_counts
from st_explore_tweet_objects import render_explore_tweet_objects
from st_featurize_tweets import render_featurize_tweets
from st_mstream_results import render_mstream_results
from st_tweet_count_api import render_tweet_count_api_interface

PAGES = {
    "MStream Results": render_mstream_results,
    "MStream Results - Inspect buckets": render_buckets,
    "Featurize tweets": render_featurize_tweets,
    "Celeb dataset": render_celeb_dataset,
    "Fetch tweet counts": render_tweet_count_api_interface,
    "Fetch tweet objects from tweet counts": render_explore_tweet_counts,
    "Inspect tweet objects": render_explore_tweet_objects,
}


PAGE_OPTIONS = list(PAGES.keys())

page_selection = st.sidebar.radio("Pages", PAGE_OPTIONS)
page = PAGES[page_selection]
page()