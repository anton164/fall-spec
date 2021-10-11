from collections import defaultdict
import numpy as np

def load_mstream_predictions(df_tweets, mstream_labels_file):
    # TODO load file
    mstream_tweetids_file = mstream_labels_file.replace("_score.txt", "_tweet_id.txt")

    tweet_id_score_map = defaultdict(lambda: 0)
    with open(mstream_tweetids_file, "r") as tweetids_f:
        with open(mstream_labels_file, "r") as labels_f:
            for (tweet_id, score) in zip(tweetids_f, labels_f):
                tweet_id_score_map[tweet_id.strip()] += float(score)

    # score_a = np.random.random(len(df_tweets))
    # score_b = np.random.random(len(df_tweets))
    # df_tweets["mstream_anomaly_score_a"] = score_a
    # df_tweets["mstream_anomaly_score_b"] = score_b

    

    df_tweets["mstream_anomaly_score"] = df_tweets["id"].apply(
        lambda tweet_id: tweet_id_score_map[tweet_id]
    )
    df_tweets["mstream_is_anomaly"] = df_tweets["mstream_anomaly_score"] > 0.5