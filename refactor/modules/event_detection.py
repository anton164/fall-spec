from math import log10
from typing import List, TypedDict, Any
from record_hash import Recordhash
from numeric_hash import Numerichash
from categoric_hash import Categorichash


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
    def __init__(
        self,
        records,
        num_buckets,
        num_rows,
        factor,
        abs_min_max,
        smoothing_factor,
        min_count,
    ) -> None:
        self.numeric_hash_tables = []
        self.categorical_hash_tables = []
        self.records = records
        self.num_buckets = num_buckets
        self.num_rows = num_rows
        self.factor = factor
        self.abs_min_max = abs_min_max
        self.smoothing_factor = smoothing_factor
        self.min_count = min_count

    def stream_data(self, preprocessed_data: List[Record]) -> List[AnomalyScores]:
        """
        The main loop that takes a slice of preprocessed data
        and returns the anomaly scores per feature
        """
        return []

    def find_max_min_numeric(self):
        max_values = []
        min_values = []
        for key in self.numeric_keys:
            max_val_key = max(self.records, key=lambda x: x[key])
            min_val_key = min(self.records, key=lambda x: x[key])
            max_values.push(max_val_key)
            min_val_key.push(min_val_key)
        return max_values, min_values

    def counts_to_anom(tot, cur, cur_t, smoothing_factor):
        # tot = total counts, no decay
        # cur = total counts with decay
        # cur_t = time elapsed
        # smoothing factor
        cur_mean = max(tot / cur_t, smoothing_factor)
        sqerr = max(0, cur - cur_mean) ** 2
        return sqerr / cur_mean + sqerr / (cur_mean * max(1, cur_t - 1))

    def stream_data(self):
        # num_rows is an unclear parameter, it could be deduced
        # from the number of entries in records. I am still not
        # sure wht is the meaning of this parameter.
        length = len(self.records)
        cur_t = 1
        dimension1 = len(self.records[0]["numerical"].keys())
        dimension2 = len(self.records[0]["categorical"].keys())
        cur_count = Recordhash(self.num_rows, self.num_buckets, dimension1, dimension2)
        total_count = Recordhash(
            self.num_rows, self.num_buckets, dimension1, dimension2
        )
        self.anom_score = []
        numeric_score = [
            Numerichash(self.num_rows, self.num_buckets) for i in range(dimension1)
        ]
        numeric_total = [
            Numerichash(self.num_rows, self.num_buckets) for i in range(dimension1)
        ]
        categ_score = [
            Categorichash(self.num_rows, self.num_buckets) for i in range(dimension2)
        ]
        categ_total = [
            Categorichash(self.num_rows, self.num_buckets) for i in range(dimension2)
        ]
        self.numeric_keys = self.records[0]["numerical"].keys().sort()
        self.categorical_keys = self.records[0]["categorical"].keys().sort()
        abs_max_numeric, abs_min_numeric = self.find_max_min_numeric()
        cur_numeric = []
        max_numeric = []
        min_numeric = []
        max_numeric_score = []
        min_numeric_score = []
        if dimension1 > 0:
            max_numeric = [float("-inf") for i in range(dimension1)]
            min_numeric = [float("inf") for i in range(dimension1)]
            max_numeric_score = [float("-inf") for i in range(dimension1)]
            min_numeric_score = [float("inf") for i in range(dimension1)]
        cur_categ = []
        for i in range(length):
            decomposed_scores = {
                key: None for key in self.numeric_keys + self.categorical_keys
            }
            decomposed_scores.score = None
            if i == 0 or self.records[i].time > cur_t:
                cur_count.lower(self.factor)
                for j in range(dimension1):
                    numeric_score[j].lower(self.factor)
                for j in range(dimension2):
                    categ_score[j].lower(self.factor)
                # implement here saving the word tokens used
            if dimension1 > 0:
                cur_numeric = [
                    self.records[i]["numerical"][key] for key in self.numeric_keys
                ]
            if dimension2 > 0:
                cur_categ = [
                    self.records[i]["categorical"][key] for key in self.categorical_keys
                ]
            sum = 0
            for node_iter in range(dimension1):
                if cur_numeric[node_iter] != 0:
                    tmp_original_numeric = cur_numeric[node_iter]
                    cur_numeric[node_iter] = log10(1 + cur_numeric[node_iter])
                    if self.abs_min_max != 1:
                        if not i:
                            max_numeric[node_iter] = cur_numeric[node_iter]
                            min_numeric[node_iter] = cur_numeric[node_iter]
                            cur_numeric[node_iter] = 0
                        else:
                            min_numeric[node_iter] = min(
                                min_numeric[node_iter], cur_numeric[node_iter]
                            )
                            max_numeric[node_iter] = max(
                                max_numeric[node_iter], cur_numeric[node_iter]
                            )
                            if max_numeric[node_iter] == min_numeric[node_iter]:
                                cur_numeric[node_iter] = 0
                            else:
                                cur_numeric[node_iter] = (
                                    cur_numeric[node_iter] - min_numeric[node_iter]
                                ) / (max_numeric[node_iter] - min_numeric[node_iter])
                    else:
                        if abs_max_numeric[node_iter] == abs_min_numeric[node_iter]:
                            cur_numeric[node_iter] = 0
                        else:
                            cur_numeric[node_iter] = (
                                cur_numeric[node_iter]
                                - log10(1 + abs_min_numeric[node_iter])
                            ) / (
                                log10(1 + abs_max_numeric[node_iter])
                                - abs_min_numeric[node_iter]
                            )
                    bucket_index = numeric_score[node_iter].hash(
                        cur_numeric[node_iter],
                        0,
                        max_numeric_score[node_iter],
                        min_numeric_score[node_iter],
                    )
                    # words_to_bucket.at(node_iter)[bucket_index].push_back(tmp_original_numeric);
                    numeric_score[node_iter].insert(
                        cur_numeric[node_iter],
                        1,
                        0,
                        max_numeric_score[node_iter],
                        min_numeric_score[node_iter],
                    )
                    numeric_total[node_iter].insert(
                        cur_numeric[node_iter],
                        1,
                        0,
                        max_numeric_score[node_iter],
                        min_numeric_score[node_iter],
                    )
                    t = self.counts_to_anom(
                        numeric_total[node_iter].get_count(
                            cur_numeric[node_iter],
                            0,
                            max_numeric_score[node_iter],
                            min_numeric_score[node_iter],
                        ),
                        numeric_score[node_iter].get_count(
                            cur_numeric[node_iter],
                            0,
                            max_numeric_score[node_iter],
                            min_numeric_score[node_iter],
                        ),
                        cur_t,
                        self.smoothing_factor,
                    )
                    min_numeric_score[node_iter] = min(min_numeric_score[node_iter], t)
                    max_numeric_score[node_iter] = max(max_numeric_score[node_iter], t)
                else:
                    t = 0
                decomposed_scores[self.numeric_keys[node_iter]] = t
                sum = sum + t
            cur_count.insert(cur_numeric, cur_categ, 1)
            total_count.insert(cur_numeric, cur_categ, 1)
            for node_iter in range(dimension2):
                tmp_original_categoric = cur_categ[node_iter]
                categ_score[node_iter].insert(cur_categ[node_iter], 1)
                categ_total[node_iter].insert(cur_categ[node_iter], 1)
                if (
                    cur_categ[node_iter] == 0
                    or categ_score[node_iter].get_count(cur_categ[node_iter])
                    <= self.min_count
                ):
                    t = 0
                else:
                    t = self.counts_to_anom(
                        categ_total[node_iter].get_count(cur_categ[node_iter]),
                        categ_score[node_iter].get_count(cur_categ[node_iter]),
                        cur_t,
                        self.smoothing_factor,
                    )
                bucket_index = categ_score[node_iter].get_bucket(tmp_original_categoric)
                decomposed_scores[self.categorical_keys[node_iter]] = t
                sum = sum+t
            if self.records[i].ignore == 0:
                cur_score = self.counts_to_anom(total_count.get_count(cur_numeric, cur_categ),
                                       cur_count.get_count(cur_numeric, cur_categ), cur_t,
                                       self.smoothing_factor)
            else:
                cur_score = 0
            sum = sum + cur_score
            decomposed_scores.score = sum
            self.anom_score.push(decomposed_scores)