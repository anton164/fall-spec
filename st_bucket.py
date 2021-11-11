from gensim.models.keyedvectors import KeyedVectors
import streamlit as st
from utils.anomaly_bucket import BucketCollection, read_buckets
from utils.st_utils import st_select_file
import json
import pandas as pd
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

@st.cache(allow_output_mutation=True)
def st_read_buckets(bucket_file, vocabulary):
    return BucketCollection(read_buckets(bucket_file, vocabulary))

@st.cache(allow_output_mutation=True)
def st_load_fasttext():
    return KeyedVectors.load_word2vec_format('./data/embeddings/fasttext/wiki-news-300d-1M.vec')

def bucket_similarities(bucket_embeddings):
    similarities = []
    for token1, emb1 in bucket_embeddings:
        for token2, emb2 in bucket_embeddings:
            if token1 != token2:
                similarities.append([token1, token2, cosine_similarity(emb1, emb2)])
    return similarities

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

    unks = sum([1 for vocab_data in vocabulary.values() if vocab_data["fasttext_idx"] == None])
    st.write(f"**Vocabulary size:** {len(vocabulary)} ({unks} unks)")

    df_vocab = pd.DataFrame(
        [(token, vocab_data["occurrences"], vocab_data["umap_representation"], vocab_data["fasttext_idx"]) 
        for token, vocab_data in vocabulary.items()], 
        columns=["token", "occurrences", "umap_value", "fasttext_idx"]
    ).sort_values("occurrences", ascending=False)
    df_vocab["fasttext_idx"] = df_vocab["fasttext_idx"].astype("Int64")
    st.write(
        df_vocab
    )

    n_buckets = buckets.size()
    n_unique_values = buckets.count_unique_values()
    utilized_buckets = buckets.count_utilized_buckets()

    st.write(f"Loaded {n_buckets} buckets ({utilized_buckets/n_buckets:.2%} utilized)")
    st.write(f"{n_unique_values} unique values hashed into {utilized_buckets} separate buckets")

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
    similarities = bucket_similarities(bucket_embeddings)
    st.subheader(f"Bucket similarities (mean: {np.mean([sim for (a, b, sim) in similarities])})")
    st.dataframe(similarities)

if __name__ == "__main__":
    render_buckets()