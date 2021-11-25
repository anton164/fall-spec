from gensim.models.keyedvectors import KeyedVectors
import streamlit as st
from utils.anomaly_bucket import BucketCollection, load_all_buckets_for_dataset
from utils.dataset import load_tweet_dataset
from utils.mstream import load_mstream_results_for_dataset
from utils.st_utils import st_select_file
import json
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
import plotly.graph_objects as go

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

@st.cache(allow_output_mutation=True)
def st_read_buckets(dataset_name, vocabulary):
    return load_all_buckets_for_dataset(
        dataset_name, 
        vocabulary
    )

@st.cache(allow_output_mutation=True)
def st_load_fasttext():
    return KeyedVectors.load_word2vec_format('./data/embeddings/fasttext/wiki-news-300d-1M.vec')

def embedding_stats(bucket_embeddings):
    tokens = [x[0] for x in bucket_embeddings]
    embeddings = [x[1] for x in bucket_embeddings]
    euclidean_distances = pairwise.euclidean_distances(
        embeddings,
        embeddings
    ).reshape(-1)
    cosine_similarities = pairwise.cosine_similarity(
        embeddings,
        embeddings
    ).reshape(-1)
    embedding_comparisons = []
    for i, (distance, similarity) in enumerate(zip(euclidean_distances, cosine_similarities)):
        token1 = tokens[int(i / len(tokens))]
        token2 = tokens[i % len(tokens)]
        if (token1 != token2):
            embedding_comparisons.append([
                token1,
                token2,
                similarity,
                distance
            ])
    
    return pd.DataFrame(
        embedding_comparisons, 
        columns=["token1", "token2", "cosine_similarity", "euclidean_distance"]
    )

def render_text_buckets(buckets, vocabulary):
    fasttext = st_load_fasttext()
    for bucket in buckets.sorted_by_collisions:
        for token in bucket.hashed_feature_values.values():
            vocabulary[token]["bucket_index"] = bucket.bucket_index

    unks = sum([1 for vocab_data in vocabulary.values() if vocab_data["fasttext_idx"] == None])
    st.write(f"**Vocabulary size:** {len(vocabulary)} ({unks} unks)")
    df_vocab = pd.DataFrame.from_dict(
        vocabulary,
        orient="index"
    ).sort_values("occurrences", ascending=False)
    df_vocab["fasttext_idx"] = df_vocab["fasttext_idx"].astype("Int64")
    df_vocab["bucket_index"] = df_vocab["bucket_index"].astype("Int64")
    st.write(
        df_vocab
    )

    fig = go.Figure()
    df_vocab_in_buckets = df_vocab[~pd.isna(df_vocab.bucket_index)].sort_values("umap_representation")
    st.subheader("Token -> Bucket visualization")
    for bucket in buckets.sorted_by_collisions:
        bucket_index = bucket.bucket_index
        df = df_vocab_in_buckets[df_vocab_in_buckets.bucket_index == bucket_index]
        if (len(df) > 0):
            fig.add_trace(
                go.Scatter(
                    y=df.umap_representation,
                    x=[bucket_index] * len(df.umap_representation),
                    text=df.index,
                    mode="markers",
                    name=f"bucket {bucket_index} ({len(df)} tokens)"
                )
            )
    fig.update_layout(
        xaxis_title="Bucket index",
        yaxis_title="UMAP Value",
    )
    st.write(fig)

    selected_bucket = st.selectbox(
        "Inspect a bucket (sorted by number of hash collisions)", 
        options=[bucket.bucket_index for bucket in buckets.sorted_by_collisions]
    )
    top_bucket = buckets.by_index[selected_bucket]
    df_bucket = pd.DataFrame(
        [(token, vocabulary[token]["occurrences"], umap_value, vocabulary[token]["fasttext_idx"]) 
        for umap_value, token in top_bucket.hashed_feature_values.items()],
        columns=["token", "occurrences", "umap_value", "fasttext_idx"]
    )
    st.write(df_bucket)
    bucket_embeddings = [
        (token, fasttext[fasttext_idx])
        for (idx, fasttext_idx, token) 
        in df_bucket[["fasttext_idx", "token"]].itertuples()
    ]
    df_embedding_stats = embedding_stats(bucket_embeddings)
    st.subheader(f"Bucket embedding comparisons")
    st.write(f"(mean similarity: {df_embedding_stats.cosine_similarity.mean()}")
    st.write(f"(mean distance: {df_embedding_stats.euclidean_distance.mean()}")
    st.dataframe(df_embedding_stats)

def render_buckets():
    st.header("Buckets")
    data_dir = "./data/labeled_datasets"
    selected_dataset = st_select_file(
        "Select dataset",
        data_dir,
        ".json"
    )
    df_tweets = load_tweet_dataset(selected_dataset)
    dataset_name = selected_dataset.replace(".json", "").replace(data_dir + "/", "")
    dataset_vocabulary = f"./MStream/data/{dataset_name}_vocabulary.json"

    with open(dataset_vocabulary, "r") as f:
        vocabulary = json.load(f)
    
    buckets_by_feature = st_read_buckets(dataset_name, vocabulary)
    selected_bucket_feature = st.selectbox(
        "Select feature to inspect",
        options=list(buckets_by_feature.keys())
    )
    buckets = buckets_by_feature[selected_bucket_feature]

    n_buckets = buckets.size()
    n_unique_values = buckets.count_unique_values()
    utilized_buckets = buckets.count_utilized_buckets()

    st.write(f"Timesteps: {buckets.total_timesteps}")
    st.write(f"Loaded {n_buckets} buckets ({utilized_buckets/n_buckets:.2%} utilized)")
    st.write(f"{n_unique_values} unique values hashed into {utilized_buckets} separate buckets")
    
    if selected_bucket_feature == "text":
        render_text_buckets(buckets, vocabulary)
    else:
        selected_bucket = st.selectbox(
            "Inspect a bucket (sorted by bucket hash frequency)", 
            options=[bucket.bucket_index for bucket in buckets.sorted_by_frequency]
        )
        top_bucket = buckets.by_index[selected_bucket]
        st.write(f"Bucket hash frequency: {top_bucket.hash_frequency()}")
        bucket_timeseries = top_bucket.timeseries(buckets.total_timesteps)
        st.write(bucket_timeseries)
        df_mstream_input, score_columns = load_mstream_results_for_dataset(
            dataset_name,
            df_tweets
        )
        st.write(df_mstream_input.timestep.max())
        
        st.subheader("Top Anomalies")
        present_score_columns = [col for col in score_columns if col in df_mstream_input.columns]
        selected_score_column = st.selectbox("By score", options=present_score_columns)
        n_tweets = st.number_input("Number of tweets to show", value=100)
        df_top_anoms = df_mstream_input.nlargest(n_tweets, selected_score_column)
        st.write(
            df_top_anoms
        )
        selected_timestep = st.number_input("Select a timestep to inspect buckets", value=0)

        active_buckets = buckets.get_buckets_by_timestep(selected_timestep)
        st.write(f"**Active buckets:** {len(active_buckets)}")
        
        for bucket in active_buckets:
            st.write(bucket.values_at_timestep(selected_timestep))

if __name__ == "__main__":
    render_buckets()