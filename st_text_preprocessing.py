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

    st.subheader("Parameters")

    col1, col2, col3 = st.columns(3)
    exclude_retweets = col1.checkbox("Exclude retweet text?", value=True)
    stem = col2.checkbox("Stem?", value=False)
    lemmatize = col3.checkbox("Lemmatize?", value=False)

    if (exclude_retweets):
        df_tweets["processed_text"] = df_tweets.apply(
            exclude_retweet_text(),
            axis=1
        )

    df_tweets["processed_text"] = df_tweets["processed_text"].apply(
        lambda t: preprocess_text(
            t,
            lemmatize=lemmatize,
            stem=stem
        ),
    ).apply(lambda t: ", ".join(t))
    
    st.table(df_tweets[["text", "processed_text"]])

if __name__ == "__main__":
    render_text_preprocessing()
