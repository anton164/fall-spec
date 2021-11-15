"""

This script converts a labeled dataset's tweet content
to its corresponding embeddings

"""


# Download fasttext embeddings and save them in data/embeddings/fasttext
# https://fasttext.cc/docs/en/english-vectors.html
import argparse
import json
from gensim.models import KeyedVectors
from utils.dataset import load_tweet_dataset
from utils.nlp import construct_vocabulary_encoding, preprocess_text
import pickle

INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './data/embeddings/'

def load_fasttext(limit):
    cache_loc = f'./data/embeddings/fasttext/cache-{limit}'
    try:
        with open(cache_loc, 'rb') as f:
            print("Loaded fasttext from cache!", cache_loc)
            return pickle.load(f)
    except Exception as e:
        print("Failed to load fasttext from cache", cache_loc)
        fasttext = KeyedVectors.load_word2vec_format('./data/embeddings/fasttext/wiki-news-300d-1M.vec', limit=limit)
        with open(cache_loc, 'wb') as f:
            print("Writing fasttext to cache", cache_loc)
            pickle.dump(fasttext, f)
        return fasttext


def tokenize_dataframe_fasttext(df, process_text=True, fasttext_limit=100000000, input_col="text"):
    if process_text:
        print("Preprocessing text...")
        df['text_tokenized'] = df[input_col].apply(lambda x: 
            preprocess_text(x)
        )
    else:
        df['text_tokenized'] = df[input_col]
    print("Loading embeddings...")
    fasttext = load_fasttext(fasttext_limit)
    print("Constructing vocabulary from dataframe...")
    return construct_vocabulary_encoding(
        df["text_tokenized"].tolist(),
        fasttext
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_file',  
        help='Input file'
    )
    parser.add_argument(
        'output_name', 
        help='Output name'
    )
    parser.add_argument(
        'fasttext_limit', 
        help='Fasttext limit',
        type=int,
        defualt=100
    )
    args = parser.parse_args()

    print("Loading tweet dataset...")
    df = load_tweet_dataset(
        INPUT_DATA_LOCATION + args.input_file
    ).set_index("id")

    vocabulary, tokenized_string_idxs, fasttext_lookup = tokenize_dataframe_fasttext(
        df,
        fasttext_limit=args.fasttext_limit
    )

    print("Writing embeddings to file...")
    with open(OUTPUT_DATA_LOCATION + args.output_name, "w") as f:
        df_dict = df.reset_index()[['id', 'text_tokenized']].to_dict(
            orient="records"
        )

        json.dump({
            "rows": df_dict,
            "fasttext_lookup": fasttext_lookup
        }, f)