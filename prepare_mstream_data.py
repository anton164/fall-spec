import pandas as pd
import argparse
from create_embeddings import tokenize_dataframe_fasttext
from utils import Timer
from utils.nlp import exclude_retweet_text, preprocess_text
from utils.dr import basic_umap_dr
from utils.dataset import load_tweet_dataset
import time
import datetime
import string
import random
from collections import defaultdict
import json
import math

INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './MStream/data/'
UNK = 0

parser = argparse.ArgumentParser()

parser.add_argument(
    '--merlion_anomaly_threshold', 
    required=False,
    default=1.5,
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
    '--text_exclude_retweets', 
    required=False,
    type=int,
    default=0,
    help='If set to 1, a tweet text will only be included once (i.e. when we process the first original tweet/retweet)'
)

parser.add_argument(
    '--text_lemmatize', 
    required=False,
    type=int,
    default=0,
    help='Whether text should be lemmatized'
)

parser.add_argument(
    '--noun_verb', 
    required=False,
    type=int,
    default=0,
    help='Whether only twets with noun verb should be considered'
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

parser.add_argument(
    '--retweet_encoding', 
    required=False,
    default="None",
    help='Type of retweet encoding [None, Categorical]'
)

parser.add_argument(
    '--mention_encoding', 
    required=False,
    default="None",
    help='Type of mention encoding [None, Categorical]'
)

parser.add_argument(
    '--window_size', 
    required=False,
    default=30,
    type=int,
    help='Window size'
)

parser.add_argument(
    '--unix_timestamp', 
    required=True,
    default=1,
    type=int,
    help='Timestamp type, if set to zero create mapping integer'
)
parser.add_argument(
    '--fasttext_limit', 
    type=int,
    required=False,
    default=100000000,
    help='Limit number of fasttext vectors'
)
parser.add_argument(
    '--downsample', 
    type=float,
    required=False,
    default=1,
    help='Downsample the dataset. 1 - retains the whole dataset , 0.5 - samples half'
)

# https://stackoverflow.com/questions/33019698/how-to-properly-round-up-half-float-numbers
# specify custom rounding method to be consistent with C
def c_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals

def map_word_to_umap(word, fasttext_dr):
    if word not in fasttext_dr:
        return UNK
    else:
        return c_round(fasttext_dr[word], 5)

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

    with Timer("Load data"):
        print("Loading tweet dataset...")
        df = load_tweet_dataset(
            INPUT_DATA_LOCATION + args.input_file
        ).set_index("id")

    if (args.downsample < 1):
        df_downsampled = df.sample(frac=args.downsample)
        print(f"Downsampled dataset, from {df.shape[0]} to {df_downsampled.shape[0]} rows")
        df = df_downsampled

    def create_unix(x):
        return int(time.mktime((x).timetuple()))
    
    def include_hashtag(hashtag):
        return args.hashtag_filter == "" or hashtag in args.hashtag_filter.split(",")

    def random_string(length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def generate_random_strings(N=1000, length=5):
        return [random_string(length) for i in range(N)]

    def sample_synthetic_string_anomaly(created_at, random_strings):
        """ Sample from random_strings, unless created_at is within some range """
        if (pd.to_datetime(created_at).day % 10 == 0):
            return "deterministic event"
        elif (pd.to_datetime(created_at).day % 5 == 0):
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
    continuous_index = []

    df["raw_text"] = df["text"]

    if (args.text_exclude_retweets):        
        df["text"] = df.apply(
            exclude_retweet_text(),
            axis=1
        )

    # Window size
    df["created_at_bucket"] = df['created_at'].dt.ceil(str(args.window_size) + 'Min')

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
    

    base_columns = ["id", "created_at_bucket", "is_anomaly"]
    extra_columns = []
    if args.text_encoding != "None":
        extra_columns.append("text")
    if args.hashtag_encoding != "None":
        extra_columns.append("hashtags")
    if args.retweet_encoding != "None":
        extra_columns.append("retweeted")
    if args.mention_encoding != "None":
        extra_columns.append("mentions")
    
    if len(extra_columns) == 0:
        processed_df = df.reset_index()[base_columns]
    else:
        processed_df = pd.DataFrame(columns = base_columns)
        for col in extra_columns:
            if col == "retweeted":
                if (args.retweet_encoding == "Categorical"):
                    df['retweeted'] = df['retweeted'].apply(
                        lambda xs: xs if xs is not None else UNK
                    )
                    print("Encoding retweets as a categorical feature...")
                    tmp_df = df.reset_index()[base_columns+[col]].explode(col)
                    for other_col in extra_columns:
                        if other_col != col:
                            tmp_df[other_col] = 0
                    processed_df = pd.concat([processed_df, tmp_df])
                    symbolic_index.append('retweeted')
            if col == "hashtags":
                if (args.hashtag_encoding == "Categorical"):
                    print("Encoding hashtags as a categorical feature...")
                    df['hashtags'] = df['hashtags'].apply(lambda xs: [x.lower() for x in xs if include_hashtag(x.lower())])
                    tmp_df = df.reset_index()[base_columns+[col]].explode(col)
                    for other_col in extra_columns:
                        if other_col != col:
                            tmp_df[other_col] = 0
                    processed_df = pd.concat([processed_df, tmp_df])
                    symbolic_index.append('hashtags')
            if col == "mentions":
                if (args.retweet_encoding == "Categorical"):
                    df['mentions'] = df['mentions'].apply(lambda xs: [x.lower() for x in xs])
                    print("Encoding mentions as a categorical feature...")
                    tmp_df = df.reset_index()[base_columns+[col]].explode(col)
                    for other_col in extra_columns:
                        if other_col != col:
                            tmp_df[other_col] = 0
                    processed_df = pd.concat([processed_df, tmp_df])
                    symbolic_index.append('mentions')
            if col == "text":
                # Text feature encoding
                df['text'] = df['text'].apply(lambda x: preprocess_text(
                    x,
                    lemmatize=args.text_lemmatize,
                    noun_verb=args.noun_verb
                ))
                tmp_df = df.reset_index()[base_columns+[col]].explode(col)
                for other_col in extra_columns:
                    if other_col != col:
                        tmp_df[other_col] = 0
                processed_df = pd.concat([processed_df, tmp_df])
                if (args.text_encoding == "Categorical"):
                    print("Encoding text as a categorical feature...")
                    symbolic_index.append('text')
                elif (args.text_encoding == "Embedding"):
                    print("Tokenizing dataframe...")
                    #raise Exception("UMAP not implemented")
                    vocabulary, tokenized_string_idxs, fasttext_lookup = tokenize_dataframe_fasttext(
                        df,
                        False,
                        fasttext_limit=args.fasttext_limit,
                        noun_verb=args.noun_verb,
                        lemmatize=args.text_lemmatize,
                    )
                    print("Vocabulary size", len(vocabulary))
                    fasttext_lookup_df = pd.DataFrame.from_dict(fasttext_lookup, orient="index").sort_index()
                    reduced_fasttext = basic_umap_dr(
                        fasttext_lookup_df, 
                        min_dist=0.1, 
                        spread=1
                    )
                    # do this step only if the dr is to 1, convert reduced fastext to dict
                    keys_list = fasttext_lookup_df.index.tolist()
                    values_list = [item for sublist in reduced_fasttext for item in sublist]
                    zip_iterator = zip(keys_list, values_list)
                    fasttext_dr = dict(zip_iterator)
                    unk_counts = defaultdict(lambda: 0)
                    processed_df['text'] = processed_df['text'].apply(map_word_to_umap, fasttext_dr=fasttext_dr)
                    for word in vocabulary.keys():
                        if word not in fasttext_dr:
                            unk_counts[word] += 1
                    continuous_index.append('text')
                    print(f"{len(unk_counts)}/{len(vocabulary)} unique unks with fasttext_limit: {args.fasttext_limit}")
                    
                    # Save vocabulary representation
                    vocabulary_dump = {}
                    for word, vocab_data in vocabulary.items():
                        vocabulary_dump[word] = {
                            "occurrences": vocab_data["occurrences"],
                            "fasttext_idx": vocab_data["fasttext_idx"],
                            "umap_representation": map_word_to_umap(word, fasttext_dr=fasttext_dr)
                        }
                        
                    with open(f"{OUTPUT_DATA_LOCATION}{args.output_name}_vocabulary.json", "w") as f:
                        json.dump(vocabulary_dump, f)
                    
    processed_df = processed_df.sort_values(by=['id', 'created_at_bucket'], ascending = (True, True))
    df_continuous = processed_df.loc[:, continuous_index]
    df_symbolic = processed_df.loc[:, symbolic_index]
    df_label = processed_df.loc[:, ['is_anomaly']]

    categorical_val_lookup = {}
    for feature in symbolic_index:
        categorical_feat_dict = {}
        for i, entry in enumerate(df_symbolic.loc[:,feature].unique()):
            categorical_feat_dict[entry] = i
        df_symbolic.loc[:, feature] =  df_symbolic.loc[:,feature].map(categorical_feat_dict)

        categorical_val_lookup[feature] = {v:k for k,v in categorical_feat_dict.items()}

    with open(f"{OUTPUT_DATA_LOCATION}{args.output_name}_categorical_val_lookup.json", "w") as f:
        json.dump(categorical_val_lookup, f)

    df[[
        "text",
        "hashtags",
        "mentions",
        "retweeted",
        "created_at",
        "is_anomaly",
        "created_at_bucket",
        "raw_text"
    ]].rename(columns={
        "text": "tokens"
    }).astype({
        "retweeted": "str"
    }).to_pickle(f"{OUTPUT_DATA_LOCATION}{args.output_name}_data.pickle")
    df_continuous.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_numeric.txt", index=False, header=False)
    df_symbolic.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_categ.txt", index=False, header=False)
    df_label.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_label.txt", index=False, header=False)

    processed_df.reset_index()[["id"]].to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_tweet_id.txt", index=False, header=False)
    processed_df.loc[:,'created_at_bucket'] = processed_df['created_at_bucket'].map(create_unix)
    if args.unix_timestamp == 0:
        timestamp_dict = {}
        for i, entry in enumerate(processed_df.loc[:,'created_at_bucket'].unique()):
            timestamp_dict[entry] = i
        processed_df.loc[:, 'created_at_bucket'] =  processed_df.loc[:,'created_at_bucket'].map(timestamp_dict)
    processed_df.loc[:,'created_at_bucket'].to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_time.txt", index=False, header=False)
    processed_df.reset_index()[["id"]].duplicated().astype(int).to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_ignore_score_record.txt", index=False, header=False)

    # save columns used
    column_names_file = open(f"{OUTPUT_DATA_LOCATION}{args.output_name}_columns.txt", "w")
    n = column_names_file.write(",".join([s + "_score" for s in ['record']+continuous_index+symbolic_index]))
    column_names_file.close()

    # Don't reeally need this anymore, unless we decide to experiment without numeric values
    '''text_file = open(f"{OUTPUT_DATA_LOCATION}{args.output_name}_numeric.txt", "w")
    df_symbolic.shape[0]
    n = text_file.write('\n'*df_symbolic.shape[0])
    text_file.close()'''