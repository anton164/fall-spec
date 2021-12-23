from modules.event_detection import EventDetection
from modules.data_transformer import DataTransformer
from modules.preprocessing import preprocess_twitter_data


if __name__ == "__main__":

    data_stream = []

    data_transformer = DataTransformer(
    )

    model = EventDetection(
        # model parameters
    )

    for raw_data in data_stream:
        clean_data = preprocess_twitter_data(raw_data)
        preprocessed_data = data_transformer.stream_data(
            clean_data
        )
        anomaly_scores = model.stream_data(
            preprocessed_data
        )
