import numpy as np

def load_mstream_predictions(df_tweets, mstream_predictions_loc):
    # TODO load file

    score_a = np.random.random(len(df_tweets))
    score_b = np.random.random(len(df_tweets))

    df_tweets["mstream_anomaly_score_a"] = score_a
    df_tweets["mstream_anomaly_score_b"] = score_b
    df_tweets["mstream_anomaly_score"] = np.mean([score_a, score_b], axis=0)
    df_tweets["mstream_is_anomaly"] = df_tweets["mstream_anomaly_score"] > 0.5