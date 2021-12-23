from typing import List, TypedDict, Any


# Dictionary of dictionary
# Record = TypedDict<{
#     "time": int,
#     "categorical": {...values},
#     "numerical": {...values}
# }>
Record = Any

# Dictionary of dictionary with anomaly scores per record
# AnomalyScores = TypedDict<{
#     "time": int,
#     "categorical": {...scores},
#     "numerical": {...scores}
# }>
AnomalyScores = Any

class EventDetection:
    def __init__(self) -> None:
        self.numeric_hash_tables = []
        self.categorical_hash_tables = []
        pass


    def stream_data(
        self, 
        preprocessed_data: List[Record]
    ) -> List[AnomalyScores]:
        """ 
        The main loop that takes a slice of preprocessed data 
        and returns the anomaly scores per feature
        """
        return []