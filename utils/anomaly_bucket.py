import math
from typing import Any, Dict
from collections import defaultdict
import sys
import pandas as pd
import json

class AnomalyBucket:
    bucket_index: int
    # mapping for (feature value -> (timestep ->  count))
    hashed_feature_value_counts_over_timesteps: Dict[Any, Dict[int, int]] 
    # mapping from feature value representation to underlying value
    hashed_feature_values: Dict[float, Any]

    def __init__(self, bucket_index) -> None:
        self.bucket_index = bucket_index
        self.hashed_feature_values = {}
        self.hashed_feature_value_counts_over_timesteps = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
    
    def hashed_value_count(self):
        return len(self.hashed_feature_values)
    
    def get_value_counts_at_timestep(self, timestep: int):
        feat_val_counts = []
        total_counter = 0
        for feat_val, count_by_timestep in self.hashed_feature_value_counts_over_timesteps.items():
            feat_val_count = count_by_timestep[timestep]
            total_counter += feat_val_count
            if (feat_val_count > 0):
                feat_val_counts.append((feat_val, self.hashed_feature_values[feat_val], feat_val_count))
        return {
            "total_count": total_counter,
            "feat_val_counts": feat_val_counts
        }
    
    def timeseries(self, n_timesteps):
        data = []
        for timestep in range(n_timesteps):
            data.append(self.get_value_counts_at_timestep(timestep))
        return pd.DataFrame(data)

class BucketCollection:
    def __init__(self, buckets_by_index: Dict[int, AnomalyBucket]) -> None:
        self.by_index = buckets_by_index
        self.sorted = sorted(
            buckets_by_index.values(), 
            key=lambda x: x.hashed_value_count(),
            reverse=True
        )

    def size(self):
        return len(self.sorted)

    def count_utilized_buckets(self):
        return sum([1 if bucket.hashed_value_count() > 0 else 0 for bucket in self.by_index.values()])
    
    def count_unique_values(self):
        return sum([bucket.hashed_value_count() for bucket in self.by_index.values()])
 
def umap_key(val):
    round_decimals = 5
    if (abs(val)) > 10:
        round_decimals -= 1
    rounded = round(val, round_decimals)
    if (float(rounded).is_integer()):
        return str(int(rounded))
    else:
        return str(rounded)
    
def read_buckets(file, vocabulary):
    umap_to_word = {
        umap_key(vocab_data["umap_representation"]): word 
        for word, vocab_data in vocabulary.items() 
        if vocab_data["umap_representation"] is not None
    }
    buckets_by_index: Dict[int, AnomalyBucket] = {}
    value_to_bucket_index = {}
    with open(file, "r") as f:
        for timestep, line in enumerate(f.readlines()):
            bucket_contents = eval(line)
            for bucket_index, content in enumerate(bucket_contents):
                if bucket_index not in buckets_by_index:
                    buckets_by_index[bucket_index] = AnomalyBucket(bucket_index)
                bucket = buckets_by_index[bucket_index]
                val_counter = defaultdict(lambda: 0)
                for val in content:
                    if val in value_to_bucket_index and bucket_index != value_to_bucket_index[val]:
                        raise Exception(f"Value {val} hashed to multiple buckets: {value_to_bucket_index[val]} and {bucket_index} at timestep {timestep}")
                    value_to_bucket_index[val] = bucket_index
                    val_counter[val] += 1
                for val, val_counter in val_counter.items():
                    bucket.hashed_feature_values[val] = umap_to_word[umap_key(val)]
                    bucket.hashed_feature_value_counts_over_timesteps[val][timestep] = val_counter
    print(f"Found {len(value_to_bucket_index)} unique values when reading bucket file {file}")
    return buckets_by_index

if __name__ == "__main__":
    buckets_file = sys.argv[1]
    vocabulary_file = buckets_file.replace("token_buckets.txt", "vocabulary.json")
    with open(vocabulary_file, "r") as f:
        vocabulary = json.load(f)
    buckets_by_index = read_buckets(buckets_file, vocabulary)
    buckets = BucketCollection(buckets_by_index)

    n_buckets = len(buckets_by_index)
    n_unique_values = buckets.count_unique_values()
    utilized_buckets = buckets.count_utilized_buckets()

    print(f"Loaded {n_buckets} buckets ({utilized_buckets/n_buckets:.2%} utilized)")
    print(f"{n_unique_values} unique values hashed into {utilized_buckets} separate buckets")
    
    print("Buckets sorted by hashed feature count")
    for bucket in buckets.sorted[:5]:
        print(f"Bucket {bucket.bucket_index} has {bucket.hashed_value_count()} unique values:")
        print(", ".join([f"{word} ({numerical})" for numerical, word in bucket.hashed_feature_values.items()]))
        print()
    top_bucket = buckets.sorted[0]
    print(f"Bucket timeseries for bucket {top_bucket.bucket_index}:")
    print(top_bucket.timeseries(100))