
from typing import List, Any, Dict
from modules.event_detection import Record
from collections import defaultdict

RawRecord = Dict[Any, Any]

# featureName -> featureVal -> encoding
FeatureEncodingLookup = Dict[
    str,
    Dict[Any, Any]
]
class FeatureEncoder:
    def __init__(self, timestep_key) -> None:
        # umap, etc.
        self.record_count = 0
        self.timestep_key = timestep_key
        self.feature_lookups: FeatureEncodingLookup = defaultdict(lambda: {})

    def encode_categorical_value(self, feature_name, val):
        if (type(val) == list):
            return [self.encode_categorical_value(feature_name, x) for x in val]
        else:
            feature_lookup = self.feature_lookups[feature_name]
            if val not in feature_lookup:
                feature_lookup[val] = len(feature_lookup) + 1
            return feature_lookup[val]

    def stream_data(
        self, 
        raw_records: List[RawRecord],
        feature_type_lookup = {},
    ) -> List[Record]:
        """ 
        Takes a slice of raw records 
        and returns records for the EventDetection
        """
        records = []
        for raw_record in raw_records:
            record: Record = {
                "time": self.record_count 
                        if self.timestep_key == "record_count" 
                        else raw_record[self.timestep_key],
                "categorical": {},
                "numerical": {},
            }
            if len(feature_type_lookup) > 0:
                for feature_name, val in raw_record.items():
                    if feature_name in feature_type_lookup:
                        feature_type = feature_type_lookup[feature_name]
                        if feature_type == "categorical":
                            record[feature_type][feature_name] = self.encode_categorical_value(feature_name, val)
                        else:
                            record[feature_type][feature_name] = val
            else:
                # infer feature type
                for feature_name, val in raw_record.items():
                    if type(val) == int or type(val) == float:
                        record["numerical"][feature_name] = val
                    else:
                        record["categorical"][feature_name] = self.encode_categorical_value(feature_name, val)
            records.append(record)
            self.record_count += 1
        return records
                