from collections import defaultdict
from os import error
import numpy as np
import pandas as pd
from utils.dataset import read_columns

def str_to_bool(s):
    if s == "True\n":
        return 1
    else:
        return 0


def load_mstream_predictions(
    df_mstream_input, 
    mstream_scores_file, 
    mstream_labels_file, 
    mstream_decomposed_scores_file, 
    mstream_decomposed_p_scores_file, 
    columns_names_file
):
    mstream_tweetids_file = mstream_scores_file.replace("_score.txt", "_tweet_id.txt")

    tweet_id_score_map = defaultdict(lambda: 0)
    tweet_id_label_map = defaultdict(lambda: 0)
    tweet_id_decomposed_score_map = defaultdict(lambda: [])
    tweet_id_decomposed_p_score_map = defaultdict(lambda: [])
    with open(mstream_tweetids_file, "r") as tweetids_f:
        with open(mstream_scores_file, "r") as scores_f:
            for (tweet_id, score) in zip(tweetids_f, scores_f):
                tweet_id_score_map[tweet_id.strip()] += float(score)
    
    with open(mstream_tweetids_file, "r") as tweetids_f:
        with open(mstream_labels_file, "r") as labels_f:
            for (tweet_id, label) in zip(tweetids_f, labels_f):
                if int(label) == 1 & tweet_id_label_map[tweet_id.strip()] != 1:
                    tweet_id_label_map[tweet_id.strip()] = 1
    
    with open(mstream_tweetids_file, "r") as tweetids_f:
        with open(mstream_decomposed_scores_file, "r") as scores_decomposed_f:
            for (tweet_id, scores_decomposed) in zip(tweetids_f, scores_decomposed_f):
                tmpValue = np.array(' '.join(scores_decomposed.split('\n')[0].split(',')).split()).astype(float)
                if len(tweet_id_decomposed_score_map[tweet_id.strip()]) == 0:
                    tweet_id_decomposed_score_map[tweet_id.strip()] = np.zeros(len(tmpValue))
                tweet_id_decomposed_score_map[tweet_id.strip()] += tmpValue
    
    with open(mstream_tweetids_file, "r") as tweetids_f:
        with open(mstream_decomposed_p_scores_file, "r") as scores_decomposed_p_f:
            for (tweet_id, scores_decomposed_p) in zip(tweetids_f, scores_decomposed_p_f):
                tmpValue = np.array(' '.join(scores_decomposed_p.split('\n')[0].split(',')).split()).astype(float)
                if len(tweet_id_decomposed_p_score_map[tweet_id.strip()]) == 0:
                    tweet_id_decomposed_p_score_map[tweet_id.strip()] = np.zeros(len(tmpValue))
                tweet_id_decomposed_p_score_map[tweet_id.strip()] += tmpValue

    df_mstream_input["mstream_anomaly_score"] = df_mstream_input["id"].apply(
        lambda tweet_id: tweet_id_score_map[tweet_id]
    )

    df_mstream_input["mstream_is_anomaly"] = df_mstream_input["id"].apply(
        lambda tweet_id: tweet_id_label_map[tweet_id]
    )
    df_mstream_input["mstream_decomposed_anomaly_score"] = df_mstream_input["id"].apply(
        lambda tweet_id: tweet_id_decomposed_score_map[tweet_id]
    )

    df_mstream_input["mstream_decomposed_p_anomaly_score"] = df_mstream_input["id"].apply(
        lambda tweet_id: tweet_id_decomposed_p_score_map[tweet_id]
    )

    df_mstream_input['mstream_decomposed_p_anomaly_score'] = df_mstream_input['mstream_decomposed_p_anomaly_score'].apply(lambda x: x/np.sum(x))
    df_mstream_input['mstream_decomposed_p_anomaly_score_tmp'] = df_mstream_input['mstream_decomposed_p_anomaly_score']*df_mstream_input['mstream_anomaly_score']
    decomposed_p_scores_columns = read_columns(columns_names_file)
    df_mstream_input[decomposed_p_scores_columns] = pd.DataFrame(df_mstream_input['mstream_decomposed_p_anomaly_score_tmp'].tolist(), index= df_mstream_input.index)

    return df_mstream_input.set_index("id")

SCORE_HANDLING_OPTIONS = {
    "unique_tweet_scores": "Unique tweet scores",
    "timestep_mean": "Timestep mean",
    "timestep_max": "Timestep max"
}

def load_mstream_results_for_dataset(dataset_name, score_handling=SCORE_HANDLING_OPTIONS["unique_tweet_scores"], mstream_data_dir="./MStream/data/"):
    mstream_scores_file = f"{mstream_data_dir}{dataset_name}_score.txt"
    mstream_labels_file = f"{mstream_data_dir}{dataset_name}_predictions.txt"
    mstream_decomposed_scores_file = f"{mstream_data_dir}{dataset_name}_decomposed.txt"
    mstream_decomposed_p_scores_file = f"{mstream_data_dir}{dataset_name}_decomposed_percentage.txt"
    columns_names_file = f"{mstream_data_dir}{dataset_name}_columns.txt"

    df_mstream_input = pd.read_pickle(f"./MStream/data/{dataset_name}_data.pickle").rename(
        columns={"text": "tokens"} # backwards-compatibility
    ).sort_values("created_at")
    df_mstream_input = load_mstream_predictions(
        df_mstream_input.reset_index(),
        mstream_scores_file,
        mstream_labels_file,
        mstream_decomposed_scores_file,
        mstream_decomposed_p_scores_file,
        columns_names_file
    )
    print(df_mstream_input.columns)
    score_columns = ['mstream_anomaly_score'] + read_columns(columns_names_file)
    df_mstream_input["is_retweet"] = df_mstream_input.retweeted.apply(
        lambda val: val is not None
    )
    if "tokens" in df_mstream_input:
        df_mstream_input["token_length"] = df_mstream_input.tokens.apply(
            lambda t: len(t)
        )
    # Score handling
    if score_handling == SCORE_HANDLING_OPTIONS["unique_tweet_scores"]:
        pass
    elif score_handling == SCORE_HANDLING_OPTIONS["timestep_mean"]:
        for score_column in score_columns:
            score_mean = df_mstream_input.groupby(['created_at_bucket'])[score_column].transform(np.mean)
            df_mstream_input[score_column] = score_mean
    elif score_handling == SCORE_HANDLING_OPTIONS["timestep_max"]:
        for score_column in score_columns:
            score_max = df_mstream_input.groupby(['created_at_bucket'])[score_column].transform(max)
            df_mstream_input[score_column] = score_max

    timestep_dict = {}
    def convert_to_timestep(val):
        if val not in timestep_dict:
            timestep_dict[val] = len(timestep_dict) + 1
        return len(timestep_dict)
    df_mstream_input["timestep"] = df_mstream_input.created_at_bucket.apply(
        lambda x: convert_to_timestep(x)
    )

    return df_mstream_input, score_columns