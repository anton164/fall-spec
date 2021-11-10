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

INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './data/embeddings/'

def tokenize_dataframe_fasttext(df, process_text=True, limit=100):
    if process_text:
        print("Preprocessing text...")
        df['text_tokenized'] = df['text'].apply(lambda x: 
            preprocess_text(x)
        )
    else:
        df['text_tokenized'] = df['text']
    print("Loading embeddings...")
    fasttext = KeyedVectors.load_word2vec_format('./data/embeddings/fasttext/wiki-news-300d-1M.vec', limit=limit)
    #fasttext = KeyedVectors.load_word2vec_format('./data/embeddings/fasttext/wiki-news-300d-1M.vec')
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
        limit=args.fasttext_limit
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