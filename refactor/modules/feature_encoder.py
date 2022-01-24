
from typing import List, Any, Dict, Union
from modules.event_detection import Record
from collections import defaultdict
import pandas as pd
from utils.dr import basic_umap_dr
from utils.nlp import construct_vocabulary_encoding, preprocess_text, load_fasttext

RANDOM_STATE = 1000
UNK = "UNK"
FASTTEXT_LIMIT = 1000000

RawRecord = Dict[Any, Any]

# featureName -> featureVal -> encoding
FeatureEncodingLookup = Dict[
    str,
    Dict[Any, Any]
]
class FeatureEncoder:
    """
        FeatureEncoder's responsibility is to stream raw records
        and emit processsed records which consist of 
        encoded categorical & numerical values
    """
    def __init__(self, timestep_key) -> None:
        # umap, etc.
        self.record_count = 0
        self.timestep_key = timestep_key
        self.feature_lookups: FeatureEncodingLookup = defaultdict(lambda: {})
        self.timestep_offset = None

    def _encode_categorical_value(self, feature_name, val):
        if (type(val) == list):
            return [self._encode_categorical_value(feature_name, x) for x in val]
        else:
            feature_lookup = self.feature_lookups[feature_name]
            if val not in feature_lookup:
                feature_lookup[val] = len(feature_lookup) + 1
            return feature_lookup[val]

    def _compute_timestep(
        self, 
        raw_record: RawRecord, 
        timestep_round: Union[str, int, None]=None # int or pd timedelta
    ):
        """ Ensure that first timestep starts at 0 """
        record_timestep = (self.record_count 
            if self.timestep_key == "record_count"
            else raw_record[self.timestep_key])
        
        if self.timestep_offset == None:
            self.timestep_offset = record_timestep
        record_timestep -= self.timestep_offset

        if isinstance(timestep_round, str):
            record_timestep = int(
                pd.Timestamp(record_timestep, unit='ms').ceil(timestep_round).timestamp() / (30 * 60)
            )
        elif isinstance(timestep_round, int):
            record_timestep = record_timestep // timestep_round
        return record_timestep
    
    def _encode_fasttext_umap(self, feature_name, val):
        tokens = preprocess_text(
            val
        )
        
        return [
            self.feature_lookups[feature_name][token] 
            if token in self.feature_lookups[feature_name] 
            else UNK
            for token in tokens
        ]

    def fit_fasttext_umap(
        self,
        raw_records: List[RawRecord],
        feature_name: str
    ):
        print(f"Preparing umap dr for '{feature_name}' feature...")
        fasttext = load_fasttext(FASTTEXT_LIMIT)
        vocabulary, tokenized_string_idxs, fasttext_lookup = construct_vocabulary_encoding(
            [preprocess_text(record[feature_name]) for record in raw_records],
            fasttext
        )
        print("Vocabulary size", len(vocabulary))
        print("Running UMAP dimensionality reduction...")
        umap_dr_result= basic_umap_dr(
            list(fasttext_lookup.values()), 
            min_dist=0.1, 
            spread=1,
            random_state=RANDOM_STATE
        )
        self.feature_lookups[feature_name] = {
            key:umap_val[0]
            for ((key,fasttext_val), umap_val)  in 
            zip(fasttext_lookup.items(), umap_dr_result)
        }

    def stream_data(
        self, 
        raw_records: List[RawRecord],
        feature_type_lookup = {},
        timestep_round: Union[int, str, None]=None # int or pd timedelta
    ) -> List[Record]:
        """ 
        Takes a slice of raw records and returns 
        encoded records for the EventDetection module
        """

        records = []
        for raw_record in raw_records:
            record: Record = {
                "time": self._compute_timestep(raw_record, timestep_round),
                "categorical": {},
                "numerical": {},
            }

            if len(feature_type_lookup) > 0:
                for feature_name, val in raw_record.items():
                    if feature_name in feature_type_lookup:
                        feature_type = feature_type_lookup[feature_name]
                        if feature_type == "fasttext_umap":
                            record["numerical"][feature_name] = self._encode_fasttext_umap(feature_name, val)
                        elif feature_type == "categorical":
                            record[feature_type][feature_name] = self._encode_categorical_value(feature_name, val)
                        else:
                            record[feature_type][feature_name] = val
            else:
                # infer feature type
                for feature_name, val in raw_record.items():
                    if type(val) == int or type(val) == float:
                        record["numerical"][feature_name] = val
                    else:
                        record["categorical"][feature_name] = self._encode_categorical_value(feature_name, val)
            records.append(record)
            self.record_count += 1
        return records
                