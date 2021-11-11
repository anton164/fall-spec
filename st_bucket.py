from gensim.models.keyedvectors import KeyedVectors
import streamlit as st
from utils.anomaly_bucket import BucketCollection, read_buckets
from utils.st_utils import st_select_file
import json
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
import plotly.graph_objects as go

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

@st.cache(allow_output_mutation=True)
def st_read_buckets(bucket_file, vocabulary):
    return BucketCollection(read_buckets(bucket_file, vocabulary))

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

def render_buckets():
    st.header("Buckets")
    data_dir = "./data/labeled_datasets"
    fasttext = st_load_fasttext()
    selected_dataset = st_select_file(
        "Select dataset",
        data_dir,
        ".json"
    )
    dataset_name = selected_dataset.replace(".json", "").replace(data_dir + "/", "")
    dataset_vocabulary = f"./MStream/data/{dataset_name}_vocabulary.json"
    dataset_token_buckets = f"./MStream/data/{dataset_name}_token_buckets.txt"

    with open(dataset_vocabulary, "r") as f:
        vocabulary = json.load(f)
    
    buckets = st_read_buckets(dataset_token_buckets, vocabulary)
    
    for bucket in buckets.sorted:
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

    n_buckets = buckets.size()
    n_unique_values = buckets.count_unique_values()
    utilized_buckets = buckets.count_utilized_buckets()

    st.write(f"Loaded {n_buckets} buckets ({utilized_buckets/n_buckets:.2%} utilized)")
    st.write(f"{n_unique_values} unique values hashed into {utilized_buckets} separate buckets")

    fig = go.Figure()
    df_vocab_in_buckets = df_vocab[~pd.isna(df_vocab.bucket_index)].sort_values("umap_representation")
    st.subheader("Bucket visualization")
    for bucket_index in df_vocab_in_buckets.bucket_index.unique():
        df = df_vocab_in_buckets[df_vocab_in_buckets.bucket_index == bucket_index]
        fig.add_trace(
            go.Scatter(
                x=df.umap_representation,
                y=df.occurrences
            )
        )
    st.write(fig)

    selected_bucket = int(st.number_input(
        "Inspect a bucket (sorted by number of values in the bucket)", 
        value=0
    ))
    top_bucket = buckets.sorted[selected_bucket]
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

if __name__ == "__main__":
    render_buckets()