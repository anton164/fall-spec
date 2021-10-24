import pandas as pd
import argparse
from create_embeddings import tokenize_dataframe_fasttext
from utils.nlp import preprocess_text
from utils.dr import basic_umap_dr
from utils.dataset import load_tweet_dataset
import time
import datetime
import string
import random


INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './MStream/data/'
UNK = -10

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

parser.add_argument(
    '--window_size', 
    required=False,
    default=30,
    type=int,
    help='Window size'
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
    ).set_index("id")[:100]

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
        if (pd.to_datetime(created_at).day % 5 == 0):
            return "this is not random"
        else:
            if random.randint(0, 1) == 1:
                return " ".join([random.choice(random_strings) for i in range(random.randint(0, 4))])
            else:
                return ""
    
    def map_word_to_umap(word, fasttext_dr):
        if word not in fasttext_dr:
            return UNK
        else:
            return fasttext_dr['word']

    # Load labels from merlion
    anomaly_threshold = args.merlion_anomaly_threshold
    df["is_anomaly"] = df["merlion_anomaly_total_count"].apply(lambda x: x > anomaly_threshold)
    df["is_anomaly_hashtag1"] = df["merlion_anomaly_top1_hashtag_count"].apply(lambda x: x > anomaly_threshold)
    df["is_anomaly_hashtag2"] = df["merlion_anomaly_top2_hashtag_count"].apply(lambda x: x > anomaly_threshold)
    df["is_anomaly_hashtag3"] = df["merlion_anomaly_top3_hashtag_count"].apply(lambda x: x > anomaly_threshold)

    symbolic_index = []
    continuous_index = []

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
    

    base_columns = ["id", "created_at", "is_anomaly"]
    extra_columns = []
    if args.text_encoding != "None":
        extra_columns.append("text")
    if args.hashtag_encoding != "None":
        extra_columns.append("hashtags")
    
    if len(extra_columns) == 0:
        processed_df = df.reset_index()[base_columns]
    else:
        processed_df = pd.DataFrame(columns = base_columns)
        for col in extra_columns:
            for other_col in extra_columns:
                if other_col != col:
                    tmp_df[other_col] = 0
            # Hashtag feature encoding
            if col == "hashtags":
                if (args.hashtag_encoding == "Categorical"):
                    print("Encoding hashtags as a categorical feature...")
                    df['hashtags'] = df['hashtags'].apply(lambda xs: [x.lower() for x in xs if include_hashtag(x.lower())])
                    tmp_df = df.reset_index()[base_columns+[col]].explode(col)
                    processed_df = pd.concat([processed_df, tmp_df])
                    symbolic_index.append('hashtags')
            if col == "text":
                # Text feature encoding
                
                df['text'] = df['text'].apply(lambda x: preprocess_text(x))
                tmp_df = df.reset_index()[base_columns+[col]].explode(col)
                processed_df = pd.concat([processed_df, tmp_df])
                print(df['text'])
                if (args.text_encoding == "Categorical"):
                    print("Encoding text as a categorical feature...")
                    symbolic_index.append('text')
                elif (args.text_encoding == "Embedding"):
                    print("Tokenizing dataframe...")
                    #raise Exception("UMAP not implemented")
                    vocabulary, tokenized_string_idxs, fasttext_lookup = tokenize_dataframe_fasttext(
                        df,
                        True
                    )
                    fasttext_lookup_df = pd.DataFrame.from_dict(fasttext_lookup, orient="index")
                    reduced_fasttext = basic_umap_dr(fasttext_lookup_df)
                    # do this step only if the dr is to 1, convert reduced fastext to dict
                    keys_list = fasttext_lookup_df.index.tolist()
                    values_list = [item for sublist in reduced_fasttext for item in sublist]
                    zip_iterator = zip(keys_list, values_list)
                    fasttext_dr = dict(zip_iterator)
                    processed_df.apply(map_word_to_umap, ar=(fasttext_dr))
                    continuous_index.append('text')
                    print(processed_df)
                    
    processed_df = processed_df.sort_values(by=['id', 'created_at'], ascending = (True, True))
    df_continuous = processed_df.loc[:, continuous_index]
    df_symbolic = processed_df.loc[:, symbolic_index]
    df_label = processed_df.loc[:, ['is_anomaly']]


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

    processed_df.reset_index()[["id"]].to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_tweet_id.txt", index=False, header=False)
    processed_df.loc[:,'created_at'] = pd.to_datetime(processed_df['created_at']).dt.floor(str(args.window_size) + 'T')
    processed_df.loc[:,'created_at'].map(create_unix).to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_time.txt", index=False, header=False)
    processed_df.reset_index()[["id"]].duplicated().astype(int).to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_ignore_score_record.txt", index=False, header=False)

    text_file = open(f"{OUTPUT_DATA_LOCATION}{args.output_name}_numeric.txt", "w")
    df_symbolic.shape[0]
    n = text_file.write('\n'*df_symbolic.shape[0])
    text_file.close()