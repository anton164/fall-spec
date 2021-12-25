
from typing import List, Any
from modules.event_detection import Record

RawRecord = Any
class DataTransformer:
    def __init__(self, timestep_key) -> None:
        # umap, etc.
        self.record_count = 0
        self.timestep_key = timestep_key


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
                for key, val in raw_record.items():
                    if key in feature_type_lookup:
                        record[feature_type_lookup[key]][key] = val
            else:
                # infer feature type
                for key, val in raw_record.items():
                    if type(val) == int or type(val) == float:
                        record["numerical"][key] = val
                    else:
                        record["categorical"][key] = val
            records.append(record)
            self.record_count += 1
        return records
                