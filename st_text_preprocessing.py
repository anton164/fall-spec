import streamlit as st
from create_embeddings import tokenize_dataframe_fasttext
from prepare_mstream_data import map_word_to_umap
from utils.dataset import load_tweet_dataset, load_vocabulary_df
from utils.dr import basic_umap_dr
from utils.nlp import exclude_retweet_text, preprocess_text
import pandas as pd
from utils.st_utils import st_select_file

@st.cache(allow_output_mutation=True)
def convert_tokens_to_numerical_feature(df_tweets, text_col, output_col):
    vocabulary, tokenized_string_idxs, fasttext_lookup = tokenize_dataframe_fasttext(
        df_tweets,
        False,
        input_col=text_col
    )
    fasttext_lookup_df = pd.DataFrame.from_dict(fasttext_lookup, orient="index")
    reduced_fasttext = basic_umap_dr(
        fasttext_lookup_df, 
        min_dist=0.1, 
        spread=1
    )
    keys_list = fasttext_lookup_df.index.tolist()
    values_list = [item for sublist in reduced_fasttext for item in sublist]
    zip_iterator = zip(keys_list, values_list)
    fasttext_dr = dict(zip_iterator)

    df_tweets[output_col] = df_tweets[text_col].apply(
        lambda tokens: [map_word_to_umap(token, fasttext_dr=fasttext_dr) for token in tokens], 
    )

    for word in vocabulary.keys():
        vocabulary[word]["umap_representation"] = map_word_to_umap(word, fasttext_dr=fasttext_dr)

    return vocabulary

def render_text_preprocessing():
    st.header("Text preprocessing")

    data_dir = "./data/labeled_datasets"
    selected_dataset = st_select_file(
        "Select dataset",
        data_dir,
        ".json"
    )
    dataset_name = selected_dataset.replace(".json", "").replace(data_dir + "/", "")
    df_tweets = load_tweet_dataset(selected_dataset)

    st.subheader("Parameters")

    col1, col2, col3 = st.columns(3)
    exclude_retweets = col1.checkbox("Exclude retweet text?", value=True)
    stem = col2.checkbox("Stem?", value=False)
    lemmatize = col3.checkbox("Lemmatize?", value=False)

    if (exclude_retweets):
        df_tweets["tokenized_text"] = df_tweets.apply(
            exclude_retweet_text(),
            axis=1
        )

    df_tweets["tokenized_text"] = df_tweets["tokenized_text"].apply(
        lambda t: preprocess_text(
            t,
            lemmatize=lemmatize,
            stem=stem
        ),
    )
    df_tweets["tokenized_text_str"] = df_tweets["tokenized_text"].apply(lambda t: ", ".join(t))
    
    with st.expander("Show first 100 preprocessed tweets"):
        st.table(df_tweets[:100][["text", "tokenized_text"]])

    st.header("How does it translate to UMAP?")

    vocabulary = convert_tokens_to_numerical_feature(
        df_tweets, 
        "tokenized_text",
        "umap_representation"
    )
    unk_count = sum([1 for vocab_data in vocabulary.values() if pd.isna(vocab_data["fasttext_idx"])])
    st.write(f"**Vocabulary size:** {len(vocabulary)} ({unk_count} unks)")
    df_vocab = pd.DataFrame.from_dict(
        vocabulary,
        orient="index"
    ).sort_values("occurrences", ascending=False)
    df_vocab["fasttext_idx"] = df_vocab["fasttext_idx"].astype("Int64")

    st.write(df_vocab)

if __name__ == "__main__":
    render_text_preprocessing()
