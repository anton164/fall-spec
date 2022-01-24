from modules.event_detection import EventDetection
from modules.feature_encoder import FeatureEncoder
from tqdm import tqdm
import ujson

def load_tweets(location, N):
    with open(location, "r") as f:
        tweets = ujson.load(f)
    yield tweets[:N]

if __name__ == "__main__":

    feature_encoder = FeatureEncoder(
        timestep_key="created_at"
    )

    model = EventDetection(
        # model parameters
    )
    
    feature_type_lookup = {
        "text": "fasttext_umap",
        "hashtags": "categorical",
        "retweeted": "categorical",
    }
    for tweet_rows in tqdm(load_tweets(
        "../data/labeled_datasets/CentralParkNYC-2021-01-27-2021-02-06.json", 
        1000
    )):
        feature_encoder.prepare_umap_dr(
            tweet_rows,
            "text"
        )
        preprocessed_data = feature_encoder.stream_data(
            tweet_rows,
            feature_type_lookup,
            timestep_round="30Min"
        )
        print(preprocessed_data)
        anomaly_scores = model.stream_data(
            preprocessed_data
        )
