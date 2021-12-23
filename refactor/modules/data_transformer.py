
from typing import List
from modules.event_detection import Record


class DataTransformer:
    def __init__(self) -> None:
        # umap, etc.
        pass


    def stream_data(self, clean_data) -> List[Record]:
        """ 
        Takes a slice of clean data 
        and returns input for the EventDetection
        """
        return []