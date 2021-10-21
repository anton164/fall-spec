import pandas as pd
import argparse
from create_embeddings import tokenize_dataframe_fasttext
from utils.nlp import preprocess_text
from utils.dataset import load_tweet_dataset
import time
import datetime
import string
import random

INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './MStream/data/'

parser = argparse.ArgumentParser()

parser.add_argument(
    '--merlion_anomaly_threshold', 
    required=False,
    default=1,
    type=float,
    help='Anomaly threshold for merlion labels'
)

parser.add_argument(
    '--text_synthetic', 
    required=False,
    default=0,
    type=int,
    help='Synthetically generate semantic text anomalies (N=random strings to smaple from, N=0 to skip)'
)

parser.add_argument(
    '--text_encoding', 
    required=False,
    default="None",
    help='Type of text encoding [None, Categorical, Embedding]'
)

parser.add_argument(
    '--hashtag_encoding', 
    required=False,
    default="None",
    help='Type of hashtag encoding [None, Categorical]'
)

parser.add_argument(
    '--hashtag_filter', 
    required=False,
    default="",
    help='Comma-separated hashtag filter'
)

if __name__ == "__main__":
    parser.add_argument(
        'input_file',  
        help='Input file'
    )
    parser.add_argument(
        'output_name', 
        help='Outuput name'
    )
    args = parser.parse_args()

    df = load_tweet_dataset(
        INPUT_DATA_LOCATION + args.input_file
    ).set_index("id")
    # TODO this is ugly don't kill me

    def create_unix(x):
        return int(time.mktime((x).timetuple()))

    def random_string(length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def generate_random_strings(N=1000, length=5):
        return [random_string(length) for i in range(N)]

    def sample_synthetic_string_anomaly(created_at, random_strings):
        """ Sample from random_strings, unless created_at is within some range """
        if (pd.to_datetime(created_at).day % 5 == 0):
            return "this is not random"
        else:
            if random.randint(0, 1) == 1:
                return " ".join([random.choice(random_strings) for i in range(random.randint(0, 4))])
            else:
                return ""

    # Load labels from merlion
    anomaly_threshold = args.merlion_anomaly_threshold
    df["is_anomaly"] = df["merlion_anomaly_total_count"].apply(lambda x: x > anomaly_threshold)
    df["is_anomaly_hashtag1"] = df["merlion_anomaly_top1_hashtag_count"].apply(lambda x: x > anomaly_threshold)
    df["is_anomaly_hashtag2"] = df["merlion_anomaly_top2_hashtag_count"].apply(lambda x: x > anomaly_threshold)
    df["is_anomaly_hashtag3"] = df["merlion_anomaly_top3_hashtag_count"].apply(lambda x: x > anomaly_threshold)

    symbolic_index = []

    # Text parsing
    if (args.text_synthetic > 0):
        random_strings = generate_random_strings(args.text_synthetic)
        df["text"] = df.apply(
            lambda t: sample_synthetic_string_anomaly(
                t.created_at, 
                random_strings
            ),
            axis=1
        )

        print("10 sampled rows from synthetically generated text")
        print(df["text"].sample(10))

    # Text feature encoding
    if (args.text_encoding == "Categorical"):
        print("Encoding text as a categorical feature...")
        df['text'] = df['text'].apply(lambda x: preprocess_text(x))
        df = df.explode('text')
        symbolic_index.append('text')
    elif (args.text_encoding == "Embedding"):
        print("Tokenizing dataframe...")
        raise Exception("UMAP not implemented")
        vocabulary, tokenized_string_idxs, fasttext_lookup = tokenize_dataframe_fasttext(
            df
        )

    def include_hashtag(hashtag):
        return args.hashtag_filter == "" or hashtag in args.hashtag_filter.split(",")


    # Hashtag feature encoding
    if (args.hashtag_encoding == "Categorical"):
        print("Encoding hashtags as a categorical feature...")
        df['hashtags'] = df['hashtags'].apply(lambda xs: [x.lower() for x in xs if include_hashtag(x.lower())])
        df = df.explode('hashtags')
        symbolic_index.append('hashtags')

    continuous_index = []

    df_continuous = df.loc[:, continuous_index]
    df_symbolic = df.loc[:, symbolic_index]
    df_label = df.loc[:, ['is_anomaly']]


    for feature in symbolic_index:
        categorical_feat_dict = {}
        for i, entry in enumerate(df_symbolic.loc[:,feature].unique()):
            categorical_feat_dict[entry] = i

        df_symbolic.loc[:, feature] =  df_symbolic.loc[:,feature].map(categorical_feat_dict)

    df[[
        "text",
        "hashtags",
        "created_at"
    ]].to_pickle(f"{OUTPUT_DATA_LOCATION}{args.output_name}_data.pickle")
    df_continuous.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_numeric.txt", index=False, header=False)
    df_symbolic.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_categ.txt", index=False, header=False)
    df_label.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_label.txt", index=False, header=False)

    df_label.reset_index()[["id"]].to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_tweet_id.txt", index=False, header=False)
    df.loc[:,'created_at'] = pd.to_datetime(df['created_at']).dt.floor('30T')
    df.loc[:,'created_at'].map(create_unix).to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_time.txt", index=False, header=False)
    df_label.reset_index()[["id"]].duplicated().astype(int).to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_ignore_score_record.txt", index=False, header=False)

    text_file = open(f"{OUTPUT_DATA_LOCATION}{args.output_name}_numeric.txt", "w")
    df_symbolic.shape[0]
    n = text_file.write('\n'*df_symbolic.shape[0])
    text_file.close()