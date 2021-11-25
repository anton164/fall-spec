import math
from typing import Any, Dict
from collections import defaultdict
import sys
import pandas as pd
import json
import os
import re
class AnomalyBucket:
    bucket_index: int
    # mapping for (timestep -> (underlying value ->  count))
    hashed_feature_value_counts_over_timesteps: Dict[int, Dict[Any, int]] 
    # mapping from feature value representation to underlying value
    hashed_feature_value_lookup: Dict[float, Any]
    hashed_feature_value_counts: Dict[float, int]

    def __init__(self, bucket_index) -> None:
        self.bucket_index = bucket_index
        self.hashed_feature_values = {}
        self.hashed_feature_value_counts = defaultdict(lambda: 0)
        self.hashed_feature_value_counts_over_timesteps = defaultdict(
            lambda: defaultdict(lambda: 0)
        )

    def hash_frequency(self):
        return sum(self.hashed_feature_value_counts.values())
    
    def hashed_value_count(self):
        return len(self.hashed_feature_values)

    def values_at_timestep(self, timestep):
        return self.hashed_feature_value_counts_over_timesteps[timestep]
    
    def timeseries(self, n_timesteps):
        data = []
        for timestep in range(n_timesteps):
            data.append(self.values_at_timestep(timestep))
        
        return pd.DataFrame(data).fillna(0)

class BucketCollection:
    def __init__(self, buckets_by_index: Dict[int, AnomalyBucket], total_timesteps) -> None:
        self.by_index = buckets_by_index
        self.sorted_by_collisions = sorted(
            buckets_by_index.values(), 
            key=lambda x: x.hashed_value_count(),
            reverse=True
        )
        self.sorted_by_frequency = sorted(
            buckets_by_index.values(), 
            key=lambda x: x.hash_frequency(),
            reverse=True
        )
        self.total_timesteps = total_timesteps

    def size(self):
        return len(self.sorted_by_collisions)

    def count_utilized_buckets(self):
        return sum([1 if bucket.hashed_value_count() > 0 else 0 for bucket in self.by_index.values()])
    
    def count_unique_values(self):
        return sum([bucket.hashed_value_count() for bucket in self.by_index.values()])

    def get_buckets_by_timestep(self, timestep):
        buckets_at_timestep = []
        for bucket in self.sorted_by_frequency:
            n_values = len(bucket.hashed_feature_value_counts_over_timesteps[timestep])
            if (n_values > 0):
                buckets_at_timestep.append(bucket)
        return buckets_at_timestep

def umap_key(val):
    round_decimals = 5
    if (abs(val)) > 10:
        round_decimals -= 1
    rounded = round(val, round_decimals)
    if (float(rounded).is_integer()):
        return str(int(rounded))
    else:
        return str(rounded)

def read_buckets_generic(file, map_value_to_feature):
    buckets_by_index: Dict[int, AnomalyBucket] = {}
    value_to_bucket_index = {}
    total_timesteps = 0
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
                    bucket.hashed_feature_values[val] = map_value_to_feature(val)
                    prev_val_count = bucket.hashed_feature_value_counts[val]
                    bucket.hashed_feature_value_counts[val] = val_counter
                    val_count_at_timestep = val_counter - prev_val_count
                    if val_count_at_timestep > 0:
                        bucket.hashed_feature_value_counts_over_timesteps[timestep][map_value_to_feature(val)] = val_count_at_timestep
            total_timesteps = timestep
        total_timesteps += 1 # because it starts at 0
    print(f"Found {len(value_to_bucket_index)} unique values when reading bucket file {file}")
    return BucketCollection(buckets_by_index, total_timesteps)

def load_bucket_files(dataset_name, data_dir="Mstream/data"):
    return [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith(dataset_name) and "buckets" in f
    ]

def load_all_buckets_for_dataset(dataset_name, vocabulary, data_dir="Mstream/data"):
    bucket_files = load_bucket_files(dataset_name, data_dir)
    buckets_by_feature = {}
    with open(
        os.path.join(
            data_dir,
            f"{dataset_name}_categorical_val_lookup.json"
        ), "r") as f:
        categorical_val_lookup = json.load(f)
    for bucket_filename in bucket_files:
        matches = re.match(".+buckets_([a-z]+)_score", bucket_filename)
        if matches is not None:
            feature_name = matches[1]
            print(f"Reading buckets for feature '{feature_name}''...")
            if feature_name == "text":
                umap_to_word = {
                    umap_key(vocab_data["umap_representation"]): word 
                    for word, vocab_data in vocabulary.items() 
                    if vocab_data["umap_representation"] is not None
                }
                buckets = read_buckets_generic(
                    bucket_filename, 
                    lambda val: umap_to_word[umap_key(val)]
                )
            else:
                buckets = read_buckets_generic(
                    bucket_filename,
                    lambda val: categorical_val_lookup[feature_name][str(val)]
                )
            buckets_by_feature[feature_name] = buckets
    return buckets_by_feature

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    with open(f"Mstream/data/{dataset_name}_vocabulary.json", "r") as f:
        vocabulary = json.load(f)

    for buckets in load_all_buckets_for_dataset(dataset_name, vocabulary).values():
        n_buckets = buckets.size()
        n_unique_values = buckets.count_unique_values()
        utilized_buckets = buckets.count_utilized_buckets()

        print(f"Loaded {n_buckets} buckets ({utilized_buckets/n_buckets:.2%} utilized)")
        print(f"{n_unique_values} unique values hashed into {utilized_buckets} separate buckets")
        
        print("Buckets sorted by hash collisions")
        for bucket in buckets.sorted_by_collisions[:5]:
            print(f"Bucket {bucket.bucket_index} has {bucket.hashed_value_count()} unique values:")
            print(", ".join([f"{word} ({numerical})" for numerical, word in bucket.hashed_feature_values.items()]))
            print()
        top_bucket = buckets.sorted_by_collisions[0]
        print(f"Bucket timeseries for bucket {top_bucket.bucket_index}:")
        print(top_bucket.timeseries(100))