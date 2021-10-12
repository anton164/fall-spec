from collections import defaultdict
from os import error
import numpy as np

def str_to_bool(s):
    if s == "True\n":
        return 1
    else:
        return 0

def load_mstream_predictions(df_tweets, mstream_scores_file, mstream_labels_file):
    print("--")
    print(mstream_labels_file)
    print(mstream_scores_file)
    print("--")
    # TODO load file
    mstream_tweetids_file = mstream_scores_file.replace("_score.txt", "_tweet_id.txt")

    tweet_id_score_map = defaultdict(lambda: 0)
    tweet_id_label_map = defaultdict(lambda: 0)
    with open(mstream_tweetids_file, "r") as tweetids_f:
        with open(mstream_scores_file, "r") as scores_f:
            for (tweet_id, score) in zip(tweetids_f, scores_f):
                tweet_id_score_map[tweet_id.strip()] += float(score)
    
    with open(mstream_tweetids_file, "r") as tweetids_f:
        with open(mstream_labels_file, "r") as labels_f:
            for (tweet_id, label) in zip(tweetids_f, labels_f):
                if int(label) == 1 & tweet_id_label_map[tweet_id.strip()] != 1:
                    tweet_id_label_map[tweet_id.strip()] = 1

    # score_a = np.random.random(len(df_tweets))
    # score_b = np.random.random(len(df_tweets))
    # df_tweets["mstream_anomaly_score_a"] = score_a
    # df_tweets["mstream_anomaly_score_b"] = score_b

    

    df_tweets["mstream_anomaly_score"] = df_tweets["id"].apply(
        lambda tweet_id: tweet_id_score_map[tweet_id]
    )
    #df_tweets["mstream_is_anomaly"] = df_tweets["mstream_anomaly_score"] > 0.5
    df_tweets["mstream_is_anomaly"] = df_tweets["id"].apply(
        lambda tweet_id: tweet_id_label_map[tweet_id]
    )