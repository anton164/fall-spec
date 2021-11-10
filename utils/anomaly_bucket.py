from typing import Any, Dict
from collections import defaultdict
import sys
import pandas as pd

class AnomalyBucket:
    # mapping for (feature value -> (timestep ->  count))
    hashed_feature_value_counts_over_timesteps: Dict[Any, Dict[int, int]] 
    # mapping from id to value
    hashed_feature_values: Dict[int, Any]

    def __init__(self) -> None:
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
            feat_val_counts.append((feat_val, feat_val_count))
        return {
            "total_count": total_counter,
            "feat_val_counts": feat_val_counts
        }
    
    def timeseries(self, n_timesteps):
        data = []
        for timestep in range(n_timesteps):
            data.append(self.get_value_counts_at_timestep(timestep))
        return pd.DataFrame(data)
    
def read_buckets(file):
    buckets_by_index: Dict[int, AnomalyBucket] = defaultdict(lambda: AnomalyBucket())
    with open(file, "r") as f:
        for timestep, line in enumerate(f.readlines()):
            bucket_contents = eval(line)
            for bucket_index, content in enumerate(bucket_contents):
                bucket = buckets_by_index[bucket_index]
                val_counter = defaultdict(lambda: 0)
                for val in content:
                    val_counter[val] += 1
                for val, val_counter in val_counter.items():
                    # TODO: load actual feature value from a map
                    bucket.hashed_feature_values[val] = ""
                    bucket.hashed_feature_value_counts_over_timesteps[val][timestep] = val_counter
    
    return buckets_by_index

if __name__ == "__main__":
    buckets_by_index = read_buckets(sys.argv[1])

    n_buckets = len(buckets_by_index)
    n_unique_values = sum([bucket.hashed_value_count() for bucket in buckets_by_index.values()])
    utilized_buckets = sum([1 if bucket.hashed_value_count() > 0 else 0 for bucket in buckets_by_index.values()])
    print(f"Loaded {n_buckets} buckets ({utilized_buckets/n_buckets:.2%} utilized)")
    print(f"{n_unique_values} unique values hashed into {utilized_buckets} separate buckets")

    print("Bucket timeseries for bucket 0:")
    print(buckets_by_index[0].timeseries(100))