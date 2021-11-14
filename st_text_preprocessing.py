import streamlit as st
from utils.dataset import load_tweet_dataset
from utils.nlp import exclude_retweet_text, preprocess_text

from utils.st_utils import st_select_file

def render_text_preprocessing():
    st.header("Text preprocessing")

    data_dir = "./data/labeled_datasets"
    selected_dataset = st_select_file(
        "Select dataset",
        data_dir,
        ".json"
    )
    dataset_name = selected_dataset.replace(".json", "").replace(data_dir + "/", "")
    df_tweets = load_tweet_dataset(selected_dataset)[:100]
    
    st.table(df_tweets[["text", "hashtags", "mentions"]])

    st.subheader("Parameters")

    exclude_retweets = st.checkbox("Exclude retweet text")
    if (exclude_retweets):
        df_tweets["text"] = df_tweets.apply(
            exclude_retweet_text(),
            axis=1
        )
    
    df_tweets["text"] = df_tweets["text"].apply(
        lambda t: preprocess_text(t),
    )
    st.subheader("After preprocessing")
    st.table(df_tweets[["text", "hashtags", "mentions"]])

if __name__ == "__main__":
    render_text_preprocessing()
