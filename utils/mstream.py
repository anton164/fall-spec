from collections import defaultdict
from os import error
import numpy as np
import pandas as pd

def str_to_bool(s):
    if s == "True\n":
        return 1
    else:
        return 0
def generate_scores_columns(column_name, size):
    columns = []
    for i in range(size):
        columns.append(column_name + '_' + str(i))
    return columns

def load_mstream_predictions(df_tweets, mstream_scores_file, mstream_labels_file, mstream_decomposed_scores_file, mstream_decomposed_p_scores_file):
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

    

    df_tweets["mstream_anomaly_score"] = df_tweets["id"].apply(
        lambda tweet_id: tweet_id_score_map[tweet_id]
    )

    df_tweets["mstream_is_anomaly"] = df_tweets["id"].apply(
        lambda tweet_id: tweet_id_label_map[tweet_id]
    )
    df_tweets["mstream_decomposed_anomaly_score"] = df_tweets["id"].apply(
        lambda tweet_id: tweet_id_decomposed_score_map[tweet_id]
    )

    df_tweets["mstream_decomposed_p_anomaly_score"] = df_tweets["id"].apply(
        lambda tweet_id: tweet_id_decomposed_p_score_map[tweet_id]
    )

    decomposed_scores_columns = generate_scores_columns('mstream_decomposed_anomaly_score', len(df_tweets['mstream_decomposed_anomaly_score'].iloc[0]))
    df_tweets[decomposed_scores_columns] = pd.DataFrame(df_tweets['mstream_decomposed_anomaly_score'].tolist(), index= df_tweets.index)

    df_tweets['mstream_decomposed_p_anomaly_score'] = df_tweets['mstream_decomposed_p_anomaly_score'].apply(lambda x: x/np.sum(x))
    df_tweets['mstream_decomposed_p_anomaly_score_z'] = df_tweets['mstream_decomposed_p_anomaly_score']*df_tweets['mstream_anomaly_score']
    decomposed_p_scores_columns = generate_scores_columns('mstream_decomposed_p_anomaly_score', len(df_tweets['mstream_decomposed_anomaly_score'].iloc[0]))
    df_tweets[decomposed_p_scores_columns] = pd.DataFrame(df_tweets['mstream_decomposed_p_anomaly_score_z'].tolist(), index= df_tweets.index)

    return df_tweets.set_index("id")