import math
import numpy as np


class Numerichash:
    def __init__(self, r, b):
        self.num_rows = r
        self.num_buckets = b
        self.count = np.zeros((self.num_rows, self.num_buckets))

    def hash(self, cur_node, hacked_lsh, max, min):
        # hacked_lsh, max and min were originally used for the hacked lsh fucntion, please refer to numerichash.cpp Numerichash::hash function to learn more
        cur_node = cur_node * (self.num_buckets - 1)
        bucket = math.floor(cur_node)
        if bucket < 0:
            bucket = (bucket % self.num_buckets + self.num_buckets) % self.num_buckets
        return bucket

    def insert(self, cur_node, weight, hacked_lsh, max, min):
        bucket = hash(cur_node, hacked_lsh, max, min)
        self.count[0][bucket] += weight

    def get_count(self, cur_node, hacked_lsh, max, min):
        bucket = hash(cur_node, hacked_lsh, max, min)
        return self.count[0][bucket]

    def clear(self):
        self.count = np.zeros((self.num_rows, self.num_buckets))

    def lower(self, factor):
        self.count = self.count * factor