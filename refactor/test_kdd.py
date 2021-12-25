from modules.event_detection import EventDetection
from modules.data_transformer import DataTransformer
from modules.preprocessing import preprocess_twitter_data
from tqdm import tqdm

def load_kdd(N):
    col_names = []
    col_types = []
    with open('./data/kddcup/columns', 'r') as f:
        for line in f.readlines():
            col = line.split(": ")
            col_names.append(col[0].strip())
            col_types.append(col[1].replace(".", "").strip())
    with open('./data/kddcup/kddcup.data', 'r') as f:
        for line in f.readlines()[:N]:
            yield {
                col:float(val) if col_type == "continuous" else val
                for col, val, col_type in 
                zip(col_names, line.split(","), col_types)
            }

if __name__ == "__main__":

    data_transformer = DataTransformer(
        timestep_key="record_count"
    )

    model = EventDetection(
        # model parameters
    )

    for kdd_row in tqdm(load_kdd(1000)):
        preprocessed_data = data_transformer.stream_data(
            [kdd_row]
        )
        print(preprocessed_data)
        anomaly_scores = model.stream_data(
            preprocessed_data
        )
