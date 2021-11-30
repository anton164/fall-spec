import math
from typing import Any, Dict, Literal, Tuple, Union
from collections import defaultdict
import sys
import pandas as pd
import json
import os
import re
class AnomalyBucket:
    bucket_index: int
    # mapping for (timestep -> (underlying twitter value ->  {"count", "score"}))
    hashed_value_scores_by_timesteps: Dict[
        int, 
        Dict[
            Any, 
            Dict[Union[Literal["score"], Literal["count"]], float]
        ]
    ] 
    # mapping from MSTREAM feature value representation to underlying twitter value
    hashed_feature_value_lookup: Dict[float, Any]
    # mapping from MSTREAM feature value to total count in bucket 
    hashed_feature_value_counts: Dict[float, int]

    def __init__(self, bucket_index) -> None:
        self.bucket_index = bucket_index
        self.hashed_feature_values = {}
        self.hashed_feature_value_counts = defaultdict(lambda: 0)
        self.hashed_value_scores_by_timesteps = defaultdict(
            lambda: {}
        )

    def hash_frequency(self):
        return sum(self.hashed_feature_value_counts.values())
    
    def hashed_value_count(self):
        return len(self.hashed_feature_values)

    def values_at_timestep(self, timestep):
        return self.hashed_value_scores_by_timesteps[timestep]

def get_timeseries_from_bucket(bucket, n_timesteps):
    data = []
    for timestep in range(n_timesteps):
        for val, scores in bucket.values_at_timestep(timestep).items():
            data.append({
                "timestep": timestep,
                "value": str(val),
                "count": scores["count"],
                "score": scores["score"],
                "bucket_index": bucket.bucket_index
            })
    
    return pd.DataFrame(
        data
    ).fillna(0).set_index("timestep")
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
        buckets_at_timestep = {}
        bucket_values_at_timestep = []
        for bucket in self.sorted_by_frequency:
            n_values = len(bucket.values_at_timestep(timestep))
            if (n_values > 0):
                for val, scores in bucket.values_at_timestep(timestep).items():
                    if bucket.bucket_index in buckets_at_timestep:
                        # update
                        buckets_at_timestep[bucket.bucket_index]["values"].append(str(val))
                        buckets_at_timestep[bucket.bucket_index]["score"] = (
                            buckets_at_timestep[bucket.bucket_index]["score"] +
                            scores["score"]
                        )
                        buckets_at_timestep[bucket.bucket_index]["values_count"] = len(
                            buckets_at_timestep[bucket.bucket_index]["values"]
                        )
                    else:
                        # create
                        buckets_at_timestep[bucket.bucket_index] = {
                            "bucket_index": bucket.bucket_index,
                            "score": scores["score"],
                            "values": [str(val)],
                            "values_count": 1
                        }
                    bucket_values_at_timestep.append({
                        "bucket_index": bucket.bucket_index,
                        "values": str(val),
                        "count": scores["count"],
                        "score": scores["score"],
                    }) 
        return buckets_at_timestep.values(), bucket_values_at_timestep


def get_combined_timeseries_for_buckets(bucket_collection):
    dfs = []
    for bucket in bucket_collection.sorted_by_frequency:
        if (bucket.hash_frequency() > 0):
            df = get_timeseries_from_bucket(bucket, bucket_collection.total_timesteps)
            dfs.append(df)
        
    return pd.concat(dfs).sort_index()

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
                score_by_val: Dict[Any, float] = {}
                # Support for reading tuples
                for bucket_val in content:
                    if (type(bucket_val) == tuple):
                        val, score = bucket_val
                    else:
                        val = bucket_val
                        score = 0 # assumes score is 0 if not present 
                    
                    # Ignore unks
                    if (val == 0.0): 
                        continue
                    score_by_val[val] = score
                    if val in value_to_bucket_index and bucket_index != value_to_bucket_index[val]:
                        raise Exception(f"Value {val} hashed to multiple buckets: {value_to_bucket_index[val]} and {bucket_index} at timestep {timestep}")
                    value_to_bucket_index[val] = bucket_index
                    val_counter[val] += 1

                # in this timestep, for each value that was read in this bucket
                # save the count (number of times value was seen) and the latest score
                for val, val_counter in val_counter.items():
                    bucket.hashed_feature_values[val] = map_value_to_feature(val)
                    bucket.hashed_feature_value_counts[val] = val_counter
                    val_count_at_timestep = val_counter
                    if val_count_at_timestep > 0:
                        bucket.hashed_value_scores_by_timesteps[timestep][map_value_to_feature(val)] = {
                            "count": val_count_at_timestep,
                            "score": score_by_val[val]
                        }
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
        print(get_timeseries_from_bucket(top_bucket, 100))